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
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Otimização: Imports para Re-ranking ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker

# --- Configurações ---
VECTOR_DB_FOLDER = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3.1:8b" 
# CORREÇÃO: Aspas fechadas corretamente na mesma linha
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
async def generate_stream(query: str, model: str):
    request_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    # Chunk inicial
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    sources_text = ""
    async for chunk in rag_chain.astream({"input": query}):
        if 'answer' in chunk and chunk['answer']:
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk['answer']}, 'finish_reason': None}]})}\n\n"
        
        if 'context' in chunk:
            for doc in chunk['context']:
                src = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', '?')
                sources_text += f"\n- {src} (p. {page})"

    if sources_text:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'\n\n**Fontes:**{sources_text}'}, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

# --- Inicialização da Chain ---
@app.on_event("startup")
async def startup_event():
    global rag_chain
    print("🔄 Inicializando RAG Avançado (MMR + Re-rank)...")
    
    if not os.path.exists(VECTOR_DB_FOLDER):
        print(f"❌ Erro: Pasta '{VECTOR_DB_FOLDER}' não encontrada.")
        return

    # 1. Embeddings e Vector Store
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embedding_function)

    # 2. Retriever com MMR (Busca por Diversidade)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.6}
    )
    flashrank_client = Ranker(model_name=RERANK_MODEL_NAME, cache_dir="flashrank_cache")
    # 3. Re-ranker (Flashrank)
    compressor = FlashrankRerank(client=flashrank_client)
    
    # 4. Retriever Otimizado
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    # 5. LLM e Prompt

    system_prompt = (
        "Você é um assistente útil e preciso. "
        "Use APENAS os contextos fornecidos abaixo para responder à pergunta do usuário. "
        "Se a resposta não estiver no contexto, diga que não sabe. "
        "Responda sempre em Português do Brasil.\n\n"
        "{context}"
    )
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Construção da Chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    print("🚀 API RAG Pronta para uso!")

# --- Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG não inicializado.")

    query = request.messages[-1].content

    if request.stream:
        return StreamingResponse(generate_stream(query, request.model), media_type="text/event-stream")

    response = await rag_chain.ainvoke({"input": query})
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[Choice(index=0, message=Message(role="assistant", content=response['answer']))],
        usage=Usage()
    )