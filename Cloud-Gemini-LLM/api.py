import os
import time
import json
import pickle
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# --- Advanced RAG Imports ---
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from flashrank import Ranker

# Load environment variables
load_dotenv()

# --- Configurations ---
VECTOR_DB_FOLDER = os.getenv("VECTOR_DB_FOLDER", "vector_db")
BM25_INDEX_FILE = os.getenv("BM25_INDEX_FILE", "bm25_index.pkl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL") # Default to a solid Gemini model
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2")

# Global Variable
rag_chain = None

# --- Models (Pydantic V2) ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    
    model_config = ConfigDict(extra="ignore")

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

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    global rag_chain
    print("\n🚀 Initializing Gemini RAG (BM25 Persistent + MultiQuery + Rerank)...")
    
    if not os.path.exists(VECTOR_DB_FOLDER):
        print(f"❌ Error: Folder '{VECTOR_DB_FOLDER}' not found. Run ingest.py first.")
        yield
        return

    # 1. Embeddings e Vector Store
    print("🔹 Loading Embeddings and ChromaDB...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embedding_function)
    
    # 2. BM25 with Persistence
    keyword_retriever = None
    
    if os.path.exists(BM25_INDEX_FILE):
        print("💾 Loading BM25 index from disk...")
        try:
            with open(BM25_INDEX_FILE, "rb") as f:
                keyword_retriever = pickle.load(f)
        except Exception as e:
            print(f"⚠️ BM25 Cache Error: {e}")

    if not keyword_retriever:
        print("🏗️ Building BM25 index (this may take a while)...")
        db_data = vectorstore.get()
        if not db_data['documents']:
            print("⚠️ CRITICAL ERROR: Vector DB is EMPTY.")
            yield
            return
            
        metadatas = db_data['metadatas'] if db_data['metadatas'] else [{} for _ in db_data['documents']]
        documents = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(db_data['documents'], metadatas)
        ]
        
        keyword_retriever = BM25Retriever.from_documents(documents)
        keyword_retriever.k = 10
        
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump(keyword_retriever, f)
            print("💾 BM25 saved to disk.")

    # 3. Ensemble
    base_vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.6}
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, base_vector_retriever],
        weights=[0.4, 0.6]
    )

    # Setup LLM (Gemini)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment.")
        yield
        return

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1,
            google_api_key=api_key
        )
    except Exception as e:
        print(f"❌ Error setting up Gemini: {e}")
        yield
        return

    # 4. MultiQuery
    print("🔹 Configuring MultiQuery and Reranker...")
    mq_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente especialista em busca. "
                   "Gere 3 versões diferentes da pergunta do usuário para buscar no banco de dados. "
                   "Mantenha o idioma da pergunta original (Português). "
                   "Retorne APENAS as perguntas, uma por linha, sem numeração."),
        ("human", "{question}")
    ])

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
        prompt=mq_prompt
    )

    # 5. Rerank
    # Note: Ensure flashrank is installed and models are downloaded/cached
    flashrank_client = Ranker(model_name=RERANK_MODEL_NAME, cache_dir="flashrank_cache")
    compressor = FlashrankRerank(client=flashrank_client, top_n=15, score_threshold=0.2)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_query_retriever 
    )

    # 6. History Aware
    contextualize_q_system_prompt = (
        "Dado um histórico de chat e a última pergunta, reformule a pergunta "
        "para ser independente, se necessário. NÃO responda, apenas reformule "
        "se a pergunta depender do contexto anterior. Caso contrário, repita a pergunta."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    # 7. Final Chain
    qa_system_prompt = """
    Você é um assistente de IA especializado em recuperação de informações precisa e baseada em evidências (RAG).
    Sua única fonte de conhecimento são os fragmentos de texto fornecidos abaixo na seção "CONTEXTO".
    
    IDIOMA OBRIGATÓRIO:
    - **TODAS AS RESPOSTAS DEVEM SER EM PORTUGUÊS (PT-BR).**
    - Mesmo que o contexto ou a pergunta estejam em outro idioma, traduza a resposta final para Português.

    ESTRUTURA DOS DADOS:
    Os fragmentos de contexto podem conter cabeçalhos injetados (ex: "CONTEXTO: Documento/Seção: ...") que identificam a origem da informação. Use esses cabeçalhos para navegar entre diferentes tópicos.

    PROTOCOLOS RÍGIDOS DE RESPOSTA (ZERO ALUCINAÇÃO):

    1. **Isolamento de Entidade (CRÍTICO):**
       - Primeiro, identifique a "Entidade Alvo" da pergunta (ex: uma raça, regra, item).
       - O contexto agora possui tags como `[DOCUMENTO_PAI: NOME]`.
       - **Regra de Ouro:** Se a pergunta é sobre "Altyra", use APENAS trechos onde `DOCUMENTO_PAI` (ou TÍTULO) seja "01 Altyra", "Altyra", etc.
       - SE O TRECHO TIVER `DOCUMENTO_PAI: ASHKA` (ou outra raça), **IGNORE-O COMPLETAMENTE**, mesmo que fale sobre "Habilidades". Confie no metadado.

    2. **Fidelidade Factual:**
       - Responda APENAS com o que está escrito no CONTEXTO.
       - Se a informação não estiver no contexto, responda EXATAMENTE: "Desculpe, essa informação não consta nos documentos processados."
       - **NUNCA** invente, suponha ou use conhecimento externo.

    3. **Consistência:**
       - Se houver informações conflitantes no contexto, aponte o conflito claramente: "Há uma divergência nos documentos: a fonte X diz A, mas a fonte Y diz B."
       - Jamais tente "resolver" o conflito por conta própria.

    4. **Listagem Completa:**
       - Se solicitado a listar itens, liste TODOS que encontrar no contexto. Não resuma.

    CONTEXTO:
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    print("✅ Gemini RAG Initialized Successfully!")
    
    yield # App runs here
    
    # --- SHUTDOWN ---
    print("🛑 Shutting down API...")

app = FastAPI(title="Cloud Gemini RAG API", lifespan=lifespan)

# --- Helper Functions ---
async def generate_stream(query: str, chat_history: List, model: str):
    request_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    # Initial chunk
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    sources_text = ""
    try:
        # Note: We must pass 'chat_history' correctly
        async for chunk in rag_chain.astream({"input": query, "chat_history": chat_history}):
            if 'answer' in chunk and chunk['answer']:
                yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk['answer']}, 'finish_reason': None}]})}\n\n"
            
            if 'context' in chunk:
                for doc in chunk['context']:
                    src = doc.metadata.get('source', 'unknown')
                    page = doc.metadata.get('page', '?')
                    # Avoid duplicated sources in the footer list
                    if src not in sources_text: 
                        sources_text += f"\n- {src} (p. {page})"
    except Exception as e:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'[Error: {str(e)}]'}, 'finish_reason': None}]})}\n\n"

    # Append sources
    if sources_text:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'\n\n**Fontes:**{sources_text}'}, 'finish_reason': None}]})}\n\n"

    # Finish
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

# --- Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system is initializing...")

    raw_messages = request.messages
    query = raw_messages[-1].content
    
    # Convert Messages to LangChain format
    chat_history = []
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

    # Static Response
    response = await rag_chain.ainvoke({
        "input": query,
        "chat_history": chat_history
    })
    
    answer_content = response['answer']
    if 'context' in response:
        sources = set([f"{doc.metadata.get('source')} (p. {doc.metadata.get('page')})" for doc in response['context']])
        if sources:
            answer_content += "\n\n**Fontes:**\n- " + "\n- ".join(sources)

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[Choice(index=0, message=Message(role="assistant", content=answer_content))],
        usage=Usage()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
