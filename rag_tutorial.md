# Guia Completo: RAG Local com Python, LangChain e Ollama

Este guia contém toda a implementação necessária para rodar uma aplicação de **Retrieval-Augmented Generation (RAG)** 100% localmente no seu computador.

Utilizamos ferramentas modernas e eficientes para garantir privacidade e performance sem depender de APIs externas.

## 🛠️ Stack Tecnológica

*   **Python 3.10+** (Linguagem base)
*   **LangChain** (Orquestração do fluxo de IA)
*   **Ollama** (Servidor de LLM local)
*   **ChromaDB** (Banco de dados vetorial local e persistente)
*   **HuggingFace Embeddings** (Modelos de embedding "all-MiniLM-L6-v2")

---

## 1. Pré-requisitos & Configuração do Ollama

Antes de executar os códigos Python, você precisa garantir que o **Ollama** esteja rodando com o modelo correto.

1.  **Instale o Ollama**: Baixe em [ollama.com](https://ollama.com).
2.  **Baixe o Modelo**: Abra seu terminal e execute:
    ```bash
    ollama pull llama3.2:3b
    ```
3.  **Mantenha o Ollama Rodando**: O aplicativo desktop do Ollama geralmente fica rodando em background (ícone na bandeja do sistema). Certifique-se de que ele está ativo (porta padrão 11434).

---

## 2. Instalação das Dependências

Crie uma pasta para seu projeto e dentro dela crie um arquivo chamado `requirements.txt` com o seguinte conteúdo.

### 📄 `requirements.txt`

```text
langchain
langchain-community
langchain-core
langchain-chroma
langchain-huggingface
langchain-ollama
chromadb
pypdf
sentence-transformers
```

**Comando de Instalação:**
No terminal, dentro da pasta do projeto, execute:

```bash
pip install -r requirements.txt
```

---

## 3. Ingestão de Dados (Criando o Cérebro)

Este script lê seus PDFs, quebra o texto em pedaços menores ("chunks"), converte esses pedaços em vetores numéricos (embeddings) e salva tudo no disco.

1.  Crie uma pasta chamada `documentos` na raiz do projeto.
2.  Coloque seus arquivos `.pdf` dentro dessa pasta.
3.  Crie o arquivo `ingest.py` com o código abaixo.

### 🐍 `ingest.py`

```python
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configurações ---
DOCS_FOLDER = "documentos"
VECTOR_DB_FOLDER = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    # 1. Verifica/Cria pasta de documentos
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"⚠️  Pasta '{DOCS_FOLDER}' não existia e foi criada.")
        print(f"👉 Por favor, coloque seus arquivos PDF dentro de '{DOCS_FOLDER}' e rode o script novamente.")
        return

    # 2. Carregar Documentos (PDF e TXT)
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]
    txt_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith('.txt')]
    
    if not pdf_files and not txt_files:
        print(f"❌ Nenhum arquivo PDF ou TXT encontrado na pasta '{DOCS_FOLDER}'.")
        return

    print(f"📂 Encontrados {len(pdf_files)} PDFs e {len(txt_files)} TXTs. Iniciando processamento...")
    
    documents = []
    
    # Processar PDFs
    for pdf_file in pdf_files:
        path = os.path.join(DOCS_FOLDER, pdf_file)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
            print(f"  ✅ [PDF] Carregado: {pdf_file} ({len(docs)} páginas)")
        except Exception as e:
            print(f"  ❌ [PDF] Erro ao carregar {pdf_file}: {e}")

    # Processar TXTs
    for txt_file in txt_files:
        path = os.path.join(DOCS_FOLDER, txt_file)
        try:
            loader = TextLoader(path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            print(f"  ✅ [TXT] Carregado: {txt_file}")
        except Exception as e:
            print(f"  ❌ [TXT] Erro ao carregar {txt_file}: {e}")

    if not documents:
        print("⚠️  Nenhum documento válido carregado.")
        return

    # 3. Split (Dividir textos)
    # chunk_size=1000 garante contexto suficiente, overlap=200 mantém coesão entre chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✂️  Documentos divididos em {len(splits)} chunks.")

    # 4. Embeddings & Persistência (Chroma)
    print("🧠 Gerando embeddings e salvando no banco vetorial (Isso pode demorar um pouco)...")
    
    # Se quiser recriar o banco do zero a cada execução, descomente as linhas abaixo:
    # if os.path.exists(VECTOR_DB_FOLDER):
    #     shutil.rmtree(VECTOR_DB_FOLDER)

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Cria e persiste o banco automaticamente no diretório especificado
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=VECTOR_DB_FOLDER
    )
    
    print(f"🚀 Sucesso! Banco vetorial salvo em '{VECTOR_DB_FOLDER}'.")
    print("👉 Agora você pode executar o 'app.py' para conversar com seus documentos.")

if __name__ == "__main__":
    main()
```

**Como rodar:**
```bash
python ingest.py
```

---

## 4. O Chat (Conversando com os Dados)

Este arquivo carrega o banco de dados que criamos e inicia uma conversa interativa no terminal.

Crie o arquivo `app.py` com o código abaixo.

### 🐍 `app.py`

```python
import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configurações ---
VECTOR_DB_FOLDER = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

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

    # 3. Setup LLM (Ollama)
    try:
        llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.1,  # Baixa temperatura para reduzir alucinações
        )
    except Exception as e:
        print(f"❌ Erro ao configurar Ollama: {e}")
        print("Verifique se o Ollama está instalado e rodando.")
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
    print("🤖 CHAT RAG INICIADO - Digite 'sair' para encerrar")
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
```

## Resumo dos Passos

1.  Instale as dependências: `pip install -r requirements.txt`
2.  Coloque PDFs em `documentos/`.
3.  Processe os dados: `python ingest.py`.
4.  Converse: `python app.py`.

Aproveite sua aplicação RAG 100% local! 🚀
