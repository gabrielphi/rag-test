import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configurações ---
DOCS_FOLDER = "documentos"
VECTOR_DB_FOLDER = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def limpar_metadados(docs):
    """
    Função auxiliar para limpar o nome do arquivo no metadado 'source'.
    Transforma 'documentos/manual_beneficios_2024.pdf' em 'Manual Beneficios 2024'.
    """
    for doc in docs:
        full_path = doc.metadata.get('source', '')
        
        # 1. Pega apenas o nome do arquivo (remove a pasta)
        filename = os.path.basename(full_path)
        
        # 2. Remove a extensão (.pdf ou .txt)
        clean_name = os.path.splitext(filename)[0]
        
        # 3. Substitui underlines e hifens por espaços e deixa Título Bonito
        clean_name = clean_name.replace('_', ' ').replace('-', ' ').title()
        
        # 4. Atualiza o metadado no objeto
        doc.metadata['source'] = clean_name

def main():
    # 1. Verifica/Cria pasta de documentos
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"⚠️  Pasta '{DOCS_FOLDER}' não existia e foi criada.")
        print(f"👉 Por favor, coloque seus arquivos PDF ou TXT dentro de '{DOCS_FOLDER}' e rode o script novamente.")
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
            
            # --- NOVO: Limpeza de Metadados ---
            limpar_metadados(docs)
            # ----------------------------------
            
            documents.extend(docs)
            # Pega o nome limpo do primeiro pedaço para mostrar no print
            nome_limpo = docs[0].metadata['source']
            print(f"  ✅ [PDF] Carregado: '{nome_limpo}' ({len(docs)} páginas)")
            
        except Exception as e:
            print(f"  ❌ [PDF] Erro ao carregar {pdf_file}: {e}")

    # Processar TXTs
    for txt_file in txt_files:
        path = os.path.join(DOCS_FOLDER, txt_file)
        try:
            loader = TextLoader(path, encoding='utf-8')
            docs = loader.load()
            
            # --- NOVO: Limpeza de Metadados ---
            limpar_metadados(docs)
            # ----------------------------------
            
            documents.extend(docs)
            nome_limpo = docs[0].metadata['source']
            print(f"  ✅ [TXT] Carregado: '{nome_limpo}'")
            
        except Exception as e:
            print(f"  ❌ [TXT] Erro ao carregar {txt_file}: {e}")

    if not documents:
        print("⚠️  Nenhum documento válido carregado.")
        return

    # 3. Split (Dividir textos)
    # Quando dividimos aqui, os chunks herdam o metadata 'source' limpo que criamos acima
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✂️  Documentos divididos em {len(splits)} chunks.")

    # 4. Embeddings & Persistência (Chroma)
    print("🧠 Gerando embeddings e salvando no banco vetorial (Isso pode demorar um pouco)...")
    
    if os.path.exists(VECTOR_DB_FOLDER):
        try:
            shutil.rmtree(VECTOR_DB_FOLDER)
            print("  🗑️  Banco antigo removido para recriação limpa.")
        except Exception as e:
            print(f"  ⚠️  Não foi possível remover a pasta antiga: {e}")

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=VECTOR_DB_FOLDER
    )
    
    print(f"🚀 Sucesso! Banco vetorial salvo em '{VECTOR_DB_FOLDER}'.")
    print("👉 Agora, no seu app.py, o metadado 'source' conterá o nome limpo (ex: 'Auxilio Educacao').")

if __name__ == "__main__":
    main()