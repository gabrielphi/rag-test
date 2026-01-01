import os
import json
import hashlib
import pickle
import time
import re
import shutil
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# --- Configurations ---
DOCS_FOLDER = "documentos"
TRACKING_FILE = "controle_ingestao.json"
VECTOR_DB_FOLDER = os.getenv("VECTOR_DB_FOLDER", "vector_db")
BM25_INDEX_FILE = os.getenv("BM25_INDEX_FILE", "bm25_index.pkl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- Helpers ---

def calcular_hash_arquivo(filepath):
    """Calculates MD5 hash to detect file changes."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def carregar_controle():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {}

def salvar_controle(dados):
    with open(TRACKING_FILE, 'w') as f:
        json.dump(dados, f, indent=4)

def formatar_nome(texto):
    return texto.replace('_', ' ').replace('-', ' ').title()

def enriquecer_metadados(doc, filepath, root_folder):
    """Enriches metadata based on file path structure."""
    rel_path = os.path.relpath(filepath, root_folder)
    parts = rel_path.split(os.sep)
    
    filename = parts[-1]
    folders = parts[:-1]
    
    clean_filename = os.path.splitext(filename)[0]
    doc.metadata['source'] = formatar_nome(clean_filename)
    doc.metadata['file_path'] = rel_path
    
    # Hierarchical Category
    if len(folders) >= 1:
        doc.metadata['category'] = formatar_nome(folders[0])
    else:
        doc.metadata['category'] = "Geral"
        
    if len(folders) >= 2:
        doc.metadata['subcategory'] = formatar_nome(folders[1])
    
    doc.metadata['file_type'] = 'text' if filepath.lower().endswith(('.txt', '.md')) else 'pdf'
    
    return doc

def limpar_texto_markdown(texto):
    """Normalizes excessive whitespace."""
    if not texto:
        return ""
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()

def carregar_txt_robusto(filepath):
    """Robust text loading (UTF-8 fallback to Latin-1)."""
    try:
        loader = TextLoader(filepath, encoding='utf-8')
        docs = loader.load()
        return docs
    except Exception:
        pass

    try:
        print(f"      ⚠️ UTF-8 failed for {os.path.basename(filepath)}. Trying 'latin-1'...")
        loader = TextLoader(filepath, encoding='latin-1')
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"      ❌ Fatal error reading {os.path.basename(filepath)}: {e}")
        return []

def atualizar_cache_bm25(vectorstore):
    print("🔄 Updating BM25 index...")
    try:
        time.sleep(1) # Ensure buffer flush
        
        data = vectorstore.get() 
        texts = data['documents']
        metadatas = data['metadatas']
        
        if not texts:
            print("⚠️  Database empty. BM25 not generated.")
            return

        docs_recuperados = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)
        ]
        
        print(f"   📊 Recalculating BM25 weights for {len(docs_recuperados)} chunks...")

        bm25_retriever = BM25Retriever.from_documents(docs_recuperados)
        bm25_retriever.k = 3 
        
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        print(f"   💾 BM25 Cache saved to '{BM25_INDEX_FILE}'!")
        
    except Exception as e:
        print(f"   ❌ Error updating BM25: {e}")

def injetar_contexto_hierarquico(splits):
    """Injects hierarchical context (Headers) into the content."""
    for split in splits:
        contexto = []
        
        if 'source' in split.metadata:
            contexto.append(f"DOCUMENTO_PAI: {split.metadata['source'].upper()}")

        if "Header 1" in split.metadata:
            contexto.append(f"TÍTULO: {split.metadata['Header 1']}")
            
        if "Header 2" in split.metadata:
            contexto.append(f"SEÇÃO: {split.metadata['Header 2']}")
            
        if "Header 3" in split.metadata:
            contexto.append(f"ITEM: {split.metadata['Header 3']}")

        if contexto:
            prefixo = " | ".join(contexto)
            split.page_content = f"[{prefixo}]\n\n{split.page_content}"
            
    return splits

# --- Main Pipeline ---

def main():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"⚠️  Folder '{DOCS_FOLDER}' created. Place your files there.")
        return

    print("🧠 Loading embedding model...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    controle_atual = carregar_controle()
    arquivos_para_processar = []
    
    print(f"🔍 Scanning files in '{DOCS_FOLDER}'...")

    for root, dirs, files in os.walk(DOCS_FOLDER):
        for arquivo in files:
            if arquivo.lower().endswith(('.md', '.txt', '.pdf')):
                caminho_completo = os.path.join(root, arquivo)
                caminho_relativo = os.path.relpath(caminho_completo, DOCS_FOLDER)
                hash_atual = calcular_hash_arquivo(caminho_completo)
                
                if caminho_relativo not in controle_atual or controle_atual[caminho_relativo] != hash_atual:
                    arquivos_para_processar.append(caminho_completo)
                    controle_atual[caminho_relativo] = hash_atual

    print(f"📋 Total files to process: {len(arquivos_para_processar)}")

    print("💾 Accessing Chroma DB...")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_FOLDER, 
        embedding_function=embedding_function
    )

    if arquivos_para_processar:
        print(f"📥 Processing {len(arquivos_para_processar)} new documents...")
        
        md_docs_batch = []  # For robust splitting
        pdf_docs_batch = [] # For standard splitting

        for path in arquivos_para_processar:
            try:
                # 1. PDF Handling
                if path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(path)
                    raw_docs = loader.load()
                    
                    for doc in raw_docs:
                        enriquecer_metadados(doc, path, DOCS_FOLDER)
                        # We don't use clean_markdown here as PDF text is often messy in different ways
                        pdf_docs_batch.extend(raw_docs)
                    
                    print(f"   ✅ Processed PDF: {os.path.basename(path)}")

                # 2. Text/Markdown Handling
                else: 
                    raw_docs = carregar_txt_robusto(path)
                    if not raw_docs:
                        continue
                        
                    for doc in raw_docs:
                        doc.page_content = limpar_texto_markdown(doc.page_content)
                        enriquecer_metadados(doc, path, DOCS_FOLDER)
                    
                    md_docs_batch.extend(raw_docs)
                    print(f"   ✅ Processed Text: {os.path.basename(path)}")
                
            except Exception as e:
                print(f"   ❌ Error loading {path}: {e}")

        # --- Splitting Strategy ---
        final_chunks = []

        # Strategy A: Markdown/Text (Robust Hierarchical Split)
        if md_docs_batch:
            print("✂️  Splitting Text/Markdown (Hierarchical)...")
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
            
            md_splits = []
            for doc in md_docs_batch:
                splits = markdown_splitter.split_text(doc.page_content)
                for split in splits:
                    split.metadata.update(doc.metadata) # Inherit metadata
                
                splits = injetar_contexto_hierarquico(splits)
                md_splits.extend(splits)
            
            # Secondary recursive split for efficient chunk sizes
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,   # MATCH: Same as Self-Hosted
                chunk_overlap=300, # MATCH: Same as Self-Hosted
                separators=["\n\n", "\n", " ", ""]
            )
            final_chunks.extend(text_splitter.split_documents(md_splits))

        # Strategy B: PDF (Standard Recursive Split)
        if pdf_docs_batch:
            print("✂️  Splitting PDFs (Standard Recursive)...")
            pdf_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            final_chunks.extend(pdf_splitter.split_documents(pdf_docs_batch))

        # Save to DB
        if final_chunks:
            print(f"📊 Adding {len(final_chunks)} chunks to vector store...")
            vectorstore.add_documents(documents=final_chunks)
            salvar_controle(controle_atual) # Update cache only on success
        else:
            print("⚠️  No clean chunks generated.")

    # Update global index
    atualizar_cache_bm25(vectorstore)

if __name__ == "__main__":
    main()
