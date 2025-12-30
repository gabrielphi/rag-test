import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
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

class Message(BaseModel):
    role: str
    content: str
    
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]

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

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG Chain not initialized (Vector DB missing or API Key missing?)")

    # Extract the last user message
    last_message = request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    query = last_message.content

    try:
        # Run retrieval chain
        response = rag_chain.invoke({"input": query})
        answer = response['answer']
        
        context_docs = response.get("context", [])
        sources_text = "\n\n**Fontes:**\n"
        if context_docs:
            for doc in context_docs:
                src = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', '?')
                sources_text += f"- {src} (p. {page})\n"
            answer += sources_text
        
        return ChatCompletionResponse(
            id="chatcmpl-gemini",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=answer),
                    finish_reason="stop"
                )
            ]
        )

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
