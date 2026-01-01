import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Configurações iguais ao app.py
VECTOR_DB_FOLDER = "vector_db"
BM25_INDEX_FILE = "bm25_index.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def debug_search(query):
    print(f"\n🔎 PERGUNTA: '{query}'")
    
    # 1. Carregar Chroma
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embedding_function)
    
    # 2. Carregar BM25
    with open(BM25_INDEX_FILE, "rb") as f:
        bm25_retriever = pickle.load(f)
    
    # 3. Criar Ensemble (Simulando o app)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever(search_kwargs={"k": 5})],
        weights=[0.4, 0.6]
    )
    
    # 4. BUSCAR
    docs = ensemble_retriever.invoke(query)
    
    print(f"✅ Encontrados {len(docs)} documentos relevantes:")
    print("-" * 40)
    for i, doc in enumerate(docs):
        print(f"📄 DOC #{i+1} | Fonte: {doc.metadata.get('source')}")
        print(f"Conteúdo: {doc.page_content[:300]}...") # Mostra só os primeiros 300 chars
        print("-" * 20)

if __name__ == "__main__":
    pergunta = input("Digite sua pergunta de teste: ")
    debug_search(pergunta)