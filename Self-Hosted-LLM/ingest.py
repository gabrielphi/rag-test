import os
import json
import hashlib
import pickle
import time
import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter , MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import yaml
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- Configurações ---
DOCS_FOLDER = "documentos"
TRACKING_FILE = "controle_ingestao.json"
VECTOR_DB_FOLDER = os.getenv("VECTOR_DB_FOLDER", "vector_db")
BM25_INDEX_FILE = os.getenv("BM25_INDEX_FILE", "bm25_index.pkl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def processar_frontmatter(doc):
    """
    Extrai metadados do YAML Frontmatter (entre --- no início do arquivo).
    Move esses dados para doc.metadata e remove o cabeçalho YAML do page_content.
    """
    padrao_yaml = r'^---\n(.*?)\n---\n'
    match = re.search(padrao_yaml, doc.page_content, re.DOTALL)
    
    if match:
        yaml_content = match.group(1)
        try:
            # Carrega o YAML
            metadados_yaml = yaml.safe_load(yaml_content)
            
            # Se for um dicionário válido, atualiza os metadados do documento
            if isinstance(metadados_yaml, dict):
                # Normaliza as chaves para minúsculas para evitar confusão
                metadados_yaml = {k.lower(): v for k, v in metadados_yaml.items()}
                doc.metadata.update(metadados_yaml)
                
                # Remove o bloco YAML do texto original para limpar a leitura
                doc.page_content = doc.page_content[match.end():]
                print(f"      🏷️  Frontmatter extraído: {list(metadados_yaml.keys())}")
        except Exception as e:
            print(f"      ⚠️  Erro ao ler YAML Frontmatter: {e}")
            
    return doc

def calcular_hash_arquivo(filepath):
    """Calcula o hash MD5 de um arquivo para detectar alterações."""
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
    rel_path = os.path.relpath(filepath, root_folder)
    parts = rel_path.split(os.sep)
    
    filename = parts[-1]
    folders = parts[:-1]
    
    clean_filename = os.path.splitext(filename)[0]
    doc.metadata['source'] = formatar_nome(clean_filename)
    doc.metadata['file_path'] = rel_path
    
    if len(folders) >= 1:
        doc.metadata['category'] = formatar_nome(folders[0])
    else:
        doc.metadata['category'] = "Geral"
        
    if len(folders) >= 2:
        doc.metadata['subcategory'] = formatar_nome(folders[1])
    
    # --- GENERIC CONTEXT LOGIC ---
    # 1. Context Type = Primeira pasta (Ex: 'Marketing', 'Legal', 'Technology')
    if len(folders) > 0:
        doc.metadata['context_type'] = formatar_nome(folders[0])
    else:
        doc.metadata['context_type'] = "General"

    # 2. Entity = Nome do Arquivo (Ex: 'Non-Disclosure Agreement', 'Python Guide')
    # A menos que seja um índice.
    
    # Detecta Índice
    if filename.upper().startswith("00_") or "INDICE" in filename.upper() or "INDEX" in filename.upper():
        doc.metadata['is_index'] = True
        doc.metadata['priority'] = "high"
        doc.metadata['entity'] = "Index"
    else:
        doc.metadata['is_index'] = False
        doc.metadata['priority'] = "normal"
        # A Entidade é o próprio assunto do arquivo
        doc.metadata['entity'] = doc.metadata['source']

    return doc

# Função limpar_texto_pdf removida pois o foco agora é Markdown limpo.
def limpar_texto_markdown(texto):
    """
    Normaliza texto markdown, removendo excesso de espaços mas preservando estrutura.
    """
    if not texto:
        return ""
    # Normaliza quebras de parágrafos (3 ou mais enters viram 2)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()

def carregar_txt_robusto(filepath):
    """
    Tenta carregar TXT forçando UTF-8 primeiro (padrão moderno), 
    depois tenta encoding do Windows (latin-1) se falhar.
    """
    # 1. Tenta UTF-8 (O Padrão Ouro)
    try:
        loader = TextLoader(filepath, encoding='utf-8')
        docs = loader.load()
        return docs
    except Exception as e:
        # Se der erro, não printa nada ainda, tenta o fallback
        pass

    # 2. Tenta Latin-1 (Windows antigo / ANSI)
    try:
        print(f"      ⚠️ UTF-8 falhou para {os.path.basename(filepath)}. Tentando 'latin-1'...")
        loader = TextLoader(filepath, encoding='latin-1')
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"      ❌ Erro fatal: Não foi possível ler {os.path.basename(filepath)}.")
        return []

def atualizar_cache_bm25(vectorstore):
    print("🔄 Iniciando atualização do índice BM25...")
    try:
        # Força um pequeno delay para garantir que o Chroma gravou tudo no disco
        time.sleep(1) 
        
        data = vectorstore.get() 
        texts = data['documents']
        metadatas = data['metadatas']
        
        if not texts:
            print("⚠️  Banco vazio. BM25 não foi gerado.")
            return

        docs_recuperados = []
        for text, meta in zip(texts, metadatas):
            docs_recuperados.append(Document(page_content=text, metadata=meta))
        
        print(f"   📊 Recalculando pesos BM25 para {len(docs_recuperados)} chunks...")

        bm25_retriever = BM25Retriever.from_documents(docs_recuperados)
        bm25_retriever.k = 3 
        
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        print(f"   💾 Cache BM25 salvo com sucesso em '{BM25_INDEX_FILE}'!")
        
    except Exception as e:
        print(f"   ❌ Erro ao atualizar BM25: {e}")
def injetar_contexto_hierarquico(splits):
    """
    Pega os metadados de Header gerados pelo MarkdownSplitter e os escreve
    explicitamente no início do conteúdo do texto para diferenciar chunks similares.
    """
    for split in splits:
        contexto = []
        
        # 1. Use Generic Headers
        if 'context_type' in split.metadata:
             contexto.append(f"CTX: {split.metadata['context_type'].upper()}")

        if 'source' in split.metadata:
            contexto.append(f"DOC: {split.metadata['source'].upper()}")
        
        # 1.1 Se for INDICE, deixa explícito
        if split.metadata.get('is_index'):
             contexto.append("TYPE: INDEX/SUMMARY")

        # 1.2 Contexto Específico (Entidade)
        if split.metadata.get('entity'):
             contexto.append(f"ENTITY: {split.metadata['entity'].upper()}")

        # Se existe Header 1 
        if "Header 1" in split.metadata:
            contexto.append(f"TITLE: {split.metadata['Header 1']}")
            
        # Se existe Header 2 (Seção)
        if "Header 2" in split.metadata:
            contexto.append(f"SEÇÃO: {split.metadata['Header 2']}")
            
        # Se existe Header 3 (Sub-Seção/Detalhe)
        if "Header 3" in split.metadata:
            contexto.append(f"ITEM: {split.metadata['Header 3']}")

        # Se tiver contexto, prepomos ao texto original com uma tag forte
        if contexto:
            prefixo = " | ".join(contexto)
            # Tag [INICIO_DE_CONTEXTO] ajuda o LLM a separar meta do conteúdo real
            split.page_content = f"[{prefixo}]\n\n{split.page_content}"
            
    return splits

def formatar_chunk_com_contexto(split):
    """
    Reescreve o page_content de CADA pedacinho para garantir que 
    ele sempre comece dizendo a quem pertence.
    """
    context_header = []
        
    # --- NOVO: Prioridade para Metadados do YAML ---
    # 1. Tipo de Documento (ex: Raça, Item, Magia)
    if 'tipo_documento' in split.metadata:
        context_header.append(f"TYPE: {str(split.metadata['tipo_documento']).upper()}")
    
    # 2. Tags (Cruciais para busca semântica)
    if 'tags' in split.metadata:
        tags = split.metadata['tags']
        if isinstance(tags, list):
            tags_str = ", ".join(tags)
        else:
            tags_str = str(tags)
        context_header.append(f"TAGS: {tags_str.upper()}")

    # --- Lógica Antiga (Mantida) ---
    # Hierarquia de Pastas
    if 'context_type' in split.metadata:
        context_header.append(f"CATEGORY: {split.metadata['context_type'].upper()}")
    
    if 'source' in split.metadata:
        context_header.append(f"FILE: {split.metadata['source'].upper()}")
    
    if 'Header 1' in split.metadata: 
        context_header.append(f"TOPIC: {split.metadata['Header 1']}")
            
    if 'Header 2' in split.metadata: 
        context_header.append(f"SECTION: {split.metadata['Header 2']}")

    # Importante: Injeta a entidade se existir para disambiguação semântica
    if split.metadata.get('entity') and split.metadata.get('entity') != "Index":
        context_header.insert(0, f"[{split.metadata['entity'].upper()}]")

    # Cria um cabeçalho curto e forte
    header_str = " | ".join(context_header)
        
    # Reescreve o conteúdo: [CABEÇALHO] + Conteúdo original
    split.page_content = f"[{header_str}]\n{split.page_content}"
    return split

def main():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"⚠️  Pasta '{DOCS_FOLDER}' criada.")
        return

    print("🧠 Carregando modelo de embeddings...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    controle_atual = carregar_controle()
    arquivos_para_processar = []
    
    print(f"🔍 Escaneando estrutura de pastas em '{DOCS_FOLDER}'...")

    for root, dirs, files in os.walk(DOCS_FOLDER):
        for arquivo in files:
            if arquivo.lower().endswith(('.md', '.txt')):
                caminho_completo = os.path.join(root, arquivo)
                caminho_relativo = os.path.relpath(caminho_completo, DOCS_FOLDER)
                hash_atual = calcular_hash_arquivo(caminho_completo)
                
                if caminho_relativo not in controle_atual or controle_atual[caminho_relativo] != hash_atual:
                    arquivos_para_processar.append(caminho_completo)
                    controle_atual[caminho_relativo] = hash_atual

    print(f"📋 Total de arquivos a processar: {len(arquivos_para_processar)}")

    print("💾 Acessando Chroma DB...")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_FOLDER, 
        embedding_function=embedding_function
    )

    if arquivos_para_processar:
        print(f"📥 Processando {len(arquivos_para_processar)} novos documentos...")
        documents = []

        for path in arquivos_para_processar:
            try:
                raw_docs = []
                if path.lower().endswith('.pdf'):
                    # PDF removido conforme solicitação
                    print(f"   ⚠️  AVISO: PDF ignorado (Suporte removido): {os.path.basename(path)}")
                    continue
                else:
                    # Carrega como Texto/Markdown
                    raw_docs = carregar_txt_robusto(path)
                
                # --- CHECK DE SEGURANÇA ---
                if not raw_docs:
                    print(f"   ⚠️  AVISO: Arquivo vazio ou ilegível: {os.path.basename(path)}")
                    continue
                
                # Exibe preview para debug
                preview = raw_docs[0].page_content[:50].replace('\n', ' ')
                print(f"   📖 Lido: {os.path.basename(path)} ({len(raw_docs[0].page_content)} chars) -> '{preview}...'")

                for doc in raw_docs:
                    
                    # --- NOVO: Processa o YAML Frontmatter PRIMEIRO ---
                    # Isso extrai as tags e remove o bloco --- do texto para não atrapalhar
                    doc = processar_frontmatter(doc)

                    # Aplica a limpeza no conteúdo (Lógica existente)
                    doc.page_content = limpar_texto_markdown(doc.page_content)
                    
                    # Enriquecimento de metadados de pasta (Lógica existente)
                    enriquecer_metadados(doc, path, DOCS_FOLDER)
                
                documents.extend(raw_docs)
                print(f"   ✅ Processado: {os.path.basename(path)}")
                
            except Exception as e:
                print(f"   ❌ Erro ao carregar {path}: {e}")

        if documents:
            print("✂️  Dividindo textos (Estratégia Híbrida: Markdown + Recursivo)...")
            
            # 1. Primeiro Split: Respeitando a estrutura lógica (Cabeçalhos)
            # Isso agrupa o texto por seções reais do RPG (Lore, Stats, Habilidades)
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
            
            # Separamos documentos Markdown (TXTs novos) dos PDFs antigos
            md_docs = []
            other_docs = []
            
            for doc in documents:
                splits = markdown_splitter.split_text(doc.page_content)
                for split in splits:
                    split.metadata.update(doc.metadata) # Herda metadados do arquivo
                md_docs.extend(splits)

            # 2. Split Strategy (Adaptive)
            # Para índices, queremos chunks GIGANTES para garantir que a lista venha inteira.
            # Para conteúdo normal (Entidades), chunks médios para precisão.
            
            docs_indices = []
            docs_normais = []
            
            for doc in md_docs:
                if doc.metadata.get('is_index'):
                    docs_indices.append(doc)
                else:
                    docs_normais.append(doc)
            
            final_splits_otimizados = []

            # A. Processa Índices (Chunks de 8000 chars - quase o arquivo todo)
            if docs_indices:
                splitter_index = RecursiveCharacterTextSplitter(
                    chunk_size=8000, 
                    chunk_overlap=500,
                    separators=["\n\n", "\n", "  "]
                )
                splits_idx = splitter_index.split_documents(docs_indices)
                print(f"   📑 Índices preservados em {len(splits_idx)} chunks grandes.")
                final_splits_otimizados.extend([formatar_chunk_com_contexto(s) for s in splits_idx])

            # B. Processa Conteúdo Normal (Chunks de 1024 chars - precisão)
            if docs_normais:
                splitter_normal = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=200, 
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                splits_norm = splitter_normal.split_documents(docs_normais)
                final_splits_otimizados.extend([formatar_chunk_com_contexto(s) for s in splits_norm])

            # 3. ...E INJETA O CONTEXTO...
            # Já feito acima durante o extend

            print(f"📊 Gerados {len(final_splits_otimizados)} chunks (Índices + Conteúdo).")
            
            print(f"📥 Adicionando vetores ao banco...")
            final_splits_otimizados = filter_complex_metadata(final_splits_otimizados)
            vectorstore.add_documents(documents=final_splits_otimizados)
            salvar_controle(controle_atual)
            

    atualizar_cache_bm25(vectorstore)

if __name__ == "__main__":
    main()

