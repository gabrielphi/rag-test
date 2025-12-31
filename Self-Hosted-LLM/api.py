import os
import time
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- LangChain Imports ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage # Importante para o histórico

# --- Otimização: Imports para Re-ranking ---
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker

# --- Configurações ---
VECTOR_DB_FOLDER = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gemma3:12b" 
RERANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

app = FastAPI(title="Self-Hosted RAG API")

# Variável Global para a Chain
rag_chain = None

# --- Modelos Pydantic (Compatíveis com LibreChat/OpenAI) ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    class Config:
        extra = "ignore"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# --- Funções Auxiliares de Stream ---
async def generate_stream(query: str, chat_history: List, model: str):
    request_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    # Chunk inicial
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    sources_text = ""
    
    # Passamos o chat_history aqui
    async for chunk in rag_chain.astream({"input": query, "chat_history": chat_history}):
        if 'answer' in chunk and chunk['answer']:
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk['answer']}, 'finish_reason': None}]})}\n\n"
        
        if 'context' in chunk:
            for doc in chunk['context']:
                src = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', '?')
                # Evita duplicatas visuais
                if src not in sources_text: 
                    sources_text += f"\n- {src} (p. {page})"

    if sources_text:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'\n\n**Fontes:**{sources_text}'}, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

# --- Inicialização da Chain ---
@app.on_event("startup")
async def startup_event():
    global rag_chain
    print("🔄 Inicializando RAG Avançado (MMR + Re-rank + Histórico)...")
    
    if not os.path.exists(VECTOR_DB_FOLDER):
        print(f"❌ Erro: Pasta '{VECTOR_DB_FOLDER}' não encontrada.")
        return

    # 1. Embeddings e Vector Store
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embedding_function)
    
    db_data = vectorstore.get()
    if not db_data['documents']:
        print("⚠️  ERRO CRÍTICO: Vector DB VAZIO.")
        return

    print(f"📚 Carregando {len(db_data['documents'])} documentos para o BM25...")

    metadatas = db_data['metadatas']
    if not metadatas:
        metadatas = [{} for _ in db_data['documents']]

    documents = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(db_data['documents'], metadatas)
    ]

    # 2. Retrievers (MMR + BM25)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.6}
    )
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, base_retriever],
        weights=[0.3, 0.7]
    )

    # 3. Re-ranker
    flashrank_client = Ranker(model_name=RERANK_MODEL_NAME, cache_dir="flashrank_cache")
    compressor = FlashrankRerank(client=flashrank_client)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)

    # --- NOVO: Lógica de Histórico (History Aware) ---
    
    # Prompt para reescrever a pergunta baseada no histórico
    contextualize_q_system_prompt = (
        "Dado um histórico de chat e a última pergunta do usuário "
        "que pode fazer referência ao contexto no histórico de chat, "
        "formule uma pergunta independente que possa ser entendida "
        "sem o histórico de chat. NÃO responda à pergunta, "
        "apenas reformule-a se necessário ou retorne-a como está."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Cria o retriever que entende histórico
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    # 4. Prompt de Resposta Final
    qa_system_prompt = (
        "Você é um assistente útil e preciso. "
        "Use APENAS os contextos fornecidos abaixo para responder à pergunta do usuário. "
        "Se a resposta não estiver no contexto, diga que não sabe. "
        "Responda sempre em Português do Brasil.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"), # Inclui histórico na geração final também
        ("human", "{input}"),
    ])

    # 5. Construção da Chain Final
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    print("🚀 API RAG Pronta para uso com suporte a Histórico!")

# --- Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG não inicializado.")

    # --- Processamento do Histórico ---
    # O LibreChat envia: [user, bot, user, bot, user_current]
    # Precisamos separar o 'user_current' e converter o resto para LangChain
    
    raw_messages = request.messages
    query = raw_messages[-1].content # A última mensagem é a pergunta atual
    
    chat_history = []
    # Iteramos sobre todas as mensagens MENOS a última
    for msg in raw_messages[:-1]:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    if request.stream:
        return StreamingResponse(
            generate_stream(query, chat_history, request.model), 
            media_type="text/event-stream"
        )

    # Chamada sem stream
    response = await rag_chain.ainvoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[Choice(index=0, message=Message(role="assistant", content=response['answer']))],
        usage=Usage()
    )