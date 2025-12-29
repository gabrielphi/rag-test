import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configurações ---
VECTOR_DB_FOLDER = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# O modelo solicitado foi "gemini-2.5 Flash", mas assumindo que seja o atual "gemini-1.5-flash" ou "gemini-2.0-flash-exp".
# Caso "gemini-2.5" seja lançado, basta alterar aqui.
LLM_MODEL = "gemini-1.5-flash"

def main():
    # 1. Verificar se o banco vetorial existe
    if not os.path.exists(VECTOR_DB_FOLDER):
        print(f"❌ Erro: O diretório '{VECTOR_DB_FOLDER}' não foi encontrado.")
        print("👉 Rode o arquivo 'ingest.py' primeiro para processar seus documentos.")
        sys.exit(1)

    print("🔄 Carregando banco de dados e modelo... aguarde.")

    # 2. Setup Embeddings e Banco Vetorial
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_FOLDER,
        embedding_function=embedding_function
    )
    
    # Configura o retriever (busca os 3 trechos mais relevantes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Setup LLM (Google Gemini)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Erro: GOOGLE_API_KEY não encontrada nas variáveis de ambiente.")
        print("👉 Crie um arquivo .env com GOOGLE_API_KEY=sua_chave_aqui")
        return

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1, # Baixa temperatura para reduzir alucinações
            google_api_key=api_key 
        )
    except Exception as e:
        print(f"❌ Erro ao configurar Gemini: {e}")
        return

    # 4. Prompt Template
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

    # 5. Criar Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n" + "="*50)
    print(f"🤖 CHAT RAG INICIADO ({LLM_MODEL}) - Digite 'sair' para encerrar")
    print("="*50 + "\n")

    # 6. Loop de Chat
    while True:
        try:
            query = input("Você: ")
            if query.lower() in ["sair", "exit", "quit"]:
                print("👋 Até logo!")
                break
            if not query.strip():
                continue

            print("⏳ Pensando...", end="\r", flush=True)
            
            # Invoca a chain
            response = rag_chain.invoke({"input": query})
            
            # Limpa linha de "Pensando..."
            print(" " * 20, end="\r")

            # Exibe Resposta
            print(f"🤖 Assistente:\n{response['answer']}\n")

            # Exibe Fontes
            print("-" * 30)
            print("📚 Fontes utilizadas:")
            sources = response.get("context", [])
            if sources:
                for idx, doc in enumerate(sources):
                    source_name = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
                    page_num = doc.metadata.get('page', '?')
                    print(f"   {idx+1}. {source_name} (Pág. {page_num})")
            else:
                print("   Nenhum contexto relevante encontrado.")
            print("-" * 30 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"\n❌ Ocorreu um erro: {e}\n")

if __name__ == "__main__":
    main()
