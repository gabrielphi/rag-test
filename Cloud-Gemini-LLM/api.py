import os
import time
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# --- Configurations ---
VECTOR_DB_FOLDER = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gemini-2.5-flash"

app = FastAPI(title="Cloud Gemini RAG API")

# Global variables
rag_chain = None

# --- Models (LibreChat Compatible) ---

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    
    # LibreChat specific fields
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

    class Config:
        extra = "ignore"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

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

# --- Helper Functions ---

async def generate_stream(query: str, model: str):
    """
    Generator that creates the SSE flow for LibreChat.
    """
    request_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    # 1. Send initial chunk (Role = Assistant)
    initial_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    sources_text = ""
    
    # 2. Iterate over LangChain Stream
    async for chunk in rag_chain.astream({"input": query}):
        
        # If chunk is part of the answer
        if 'answer' in chunk and chunk['answer']:
            content = chunk['answer']
            response_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(response_chunk)}\n\n"
        
        # If chunk contains context (documents)
        if 'context' in chunk:
            for doc in chunk['context']:
                src = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', '?')
                sources_text += f"\n- {src} (p. {page})"

    # 3. Send sources as the last piece of text
    if sources_text:
        formatted_sources = f"\n\n**Fontes:**{sources_text}"
        source_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": formatted_sources}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(source_chunk)}\n\n"

    # 4. Send final closing chunk
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    
    # OpenAI official stream end signal
    yield "data: [DONE]\n\n"

# --- Initialization ---

@app.on_event("startup")
async def startup_event():
    global rag_chain
    print("🔄 Initializing Gemini RAG Chain...")
    
    if not os.path.exists(VECTOR_DB_FOLDER):
        print(f"❌ Error: Directory '{VECTOR_DB_FOLDER}' not found.")
        return

    # Setup Embeddings & Vector DB
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_FOLDER,
        embedding_function=embedding_function
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Setup LLM (Gemini)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment.")
        return

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1,
            google_api_key=api_key
        )
    except Exception as e:
        print(f"❌ Error setting up Gemini: {e}")
        return

    # Setup Prompt
    system_prompt = (
        "Você é um assistente útil e preciso. "
        "Use APENAS os contextos fornecidos abaixo para responder à pergunta do usuário. "
        "Se a resposta não estiver no contexto, diga que não sabe. "
        "Responda sempre em Português do Brasil.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ Gemini RAG Chain ready!")

# --- Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG Chain not initialized (Vector DB missing or API Key missing?)")

    # Extract the last user message
    last_message = request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    query = last_message.content

    # === STREAMING MODE ===
    if request.stream:
        print(f"🌊 Streaming request for: {query[:20]}...")
        return StreamingResponse(
            generate_stream(query, request.model),
            media_type="text/event-stream"
        )

    # === NON-STREAMING MODE (Legacy/Compatibility) ===
    print(f"📦 Static request for: {query[:20]}...")
    try:
        # Run retrieval chain
        response = await rag_chain.ainvoke({"input": query}) # Use ainvoke for async
        answer = response['answer']
        
        context_docs = response.get("context", [])
        if context_docs:
            answer += "\n\n**Fontes:**\n"
            for doc in context_docs:
                src = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', '?')
                answer += f"- {src} (p. {page})\n"
        
        # Token calculation (Estimate)
        p_tokens = len(query) // 4
        c_tokens = len(answer) // 4

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=answer),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens
            )
        )

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
