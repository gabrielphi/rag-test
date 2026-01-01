import os
import time
import json
import pickle
from contextlib import asynccontextmanager
from typing import List, Optional
import re

# --- FastAPI & Pydantic ---
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# --- LangChain Imports ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# --- Otimização: Re-ranking e MultiQuery ---
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from flashrank import Ranker
from langchain_core.output_parsers import JsonOutputParser

# --- Configurações ---
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()

# --- Constantes ---
VECTOR_DB_FOLDER = os.getenv("VECTOR_DB_FOLDER", "vector_db")
BM25_INDEX_FILE = os.getenv("BM25_INDEX_FILE", "bm25_index.pkl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2")
MAX_HISTORY_MESSAGES = 6


# --- Variáveis Globais (Componentes Reutilizáveis) ---
# Não guardamos mais a chain pronta, mas sim as peças para montá-la
vectorstore_global = None
keyword_retriever_global = None
llm_global = None
reranker_global = None
qa_prompt_global = None
contextualize_prompt_global = None
VALID_TOPICS = set() # Cache de tópicos válidos (arquivos/pastas)


# --- Modelos Pydantic ---

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

# --- NOVO: Modelo de Intenção de Busca ---
class SearchIntent(BaseModel):
    entities: List[str] = Field(
        default=[], 
        description="Nomes específicos, Projetos, Produtos, Lugares ou Arquivos. Ex: ['Projeto Alpha', 'Relatório Anual']"
    )
    topics: List[str] = Field(
        default=[], 
        description="Temas gerais, conceitos abstratos. Ex: ['Marketing', 'Segurança', 'Vendas']"
    )
    is_global_query: bool = Field(
        default=False, 
        description="True se a pergunta for genérica demais para filtrar (ex: 'Como jogar?', 'Liste todas as raças')."
    )
    # --- NOVO: Suporte a Priorização de Índices ---
    requires_index: bool = Field(
        default=False,
        description="True se o usuário pede uma LISTA, RESUMO ou VISÃO GERAL. Ex: 'Quais raças existem?', 'Resuma o sistema'."
    )
    context_filter: Optional[str] = Field(
        default=None,
        description="Filtro de contexto inferido: 'Raça', 'Classe', 'Regra' ou None."
    )
    model_config = ConfigDict(extra="ignore") # Blindagem: Ignora campos extras alucinados pelo LLM

# --- Funções de Lógica RAG ---

# --- Funções de Lógica RAG ---

def normalize_topic_name(name: str) -> str:
    """
    Normaliza nome de arquivo/pasta para comparação fuzzy.
    Ex: '00_INDICE_MUNDO.txt' -> '00 indice mundo'
    Ex: 'era_das_bestas' -> 'era das bestas'
    """
    # Remove extensão
    name = os.path.splitext(name)[0]
    # Substitui underscore e hífens por espaço
    name = name.replace("_", " ").replace("-", " ")
    return name.lower().strip()

def load_valid_topics():
    """
    Escaneia a pasta de documentos e popula o set VALID_TOPICS.
    """
    global VALID_TOPICS
    VALID_TOPICS.clear()
    
    if not os.path.exists("documentos"):
        print("⚠️ Pasta 'documentos' não encontrada para indexação de tópicos.")
        return

    print("📂 Indexando tópicos válidos...")
    count = 0
    for root, dirs, files in os.walk("documentos"):
        # Indexa nomes de pastas
        for d in dirs:
            normalized = normalize_topic_name(d)
            VALID_TOPICS.add(normalized)
            count += 1
            
        # Indexa nomes de arquivos
        for f in files:
            normalized = normalize_topic_name(f)
            VALID_TOPICS.add(normalized)
            count += 1
            
    print(f"✅ {count} tópicos válidos indexados.")

async def extract_search_intent(query: str, llm) -> SearchIntent:
    """
    Versão Blindada: Usa Regex para limpar markdown e força JSON puro.
    Resistente a 'chatice' do modelo (introduções, repetições).
    """
    # 1. Prompt muito mais explícito e autoritário
    system_prompt = """
    ATENÇÃO: Você é um SISTEMA (API), não um assistente de chat.
    Sua única função é converter a query do usuário em um objeto JSON de filtros.
    NÃO responda a pergunta. NÃO repita a pergunta. NÃO explique.
    
    SCHEMA JSON OBRIGATÓRIO:
    {{
        "entities": ["SpecificSubject", "ProperNoun", "ProjectName"],
        "topics": ["GeneralTheme", "Concept", "Process"],
        "is_global_query": boolean,
        "requires_index": boolean,
        "context_filter": "CategoryName" | null
    }}

    REGRAS DE OURO:
    1. Se a pergunta for sobre UM ASSUNTO ESPECÍFICO (Ex: 'Projeto X', 'Cliente Y', 'Arquivo Z'), coloque em 'entities'.
    2. ENTIDADE = Nomes Próprios, Projetos, Produtos, Arquivos Específicos.
    3. Se o usuário quer UMA LISTA ou VISÃO GERAL, marque "requires_index": true.
    
    EXEMPLOS:
    Input: "O que diz a politica de RH?"
    Output: {{"entities": ["Politica de RH"], "topics": [], "is_global_query": false, "requires_index": false, "context_filter": "RH"}}
    
    Input: "Como configurar o ambiente?"
    Output: {{"entities": [], "topics": ["Configurar Ambiente"], "is_global_query": false, "requires_index": false, "context_filter": "Tecnologia"}}
    
    Input: "Liste todos os documentos de marketing"
    Output: {{"entities": [], "topics": [], "is_global_query": false, "requires_index": true, "context_filter": "Marketing"}}

    Input: "Como funciona o combate?"
    Output: {{"entities": [], "topics": ["Combate"], "is_global_query": false, "requires_index": false, "context_filter": "Regra"}}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Input: {question}\nOutput JSON:")
    ])
    
    # Invocamos o LLM diretamente (sem o parser na chain para podermos limpar o texto antes)
    chain = prompt | llm
    
    try:
        response_msg = await chain.ainvoke({"question": query})
        
        # O LangChain pode retornar string ou objeto AIMessage dependendo da versão
        content = response_msg.content if hasattr(response_msg, 'content') else str(response_msg)

        # --- DEBUG CRÍTICO: Ver o que o modelo cuspiu ---
        print(f"🕵️ RAW INTENT RESPONSE: {content[:200]}...") 
        
        # --- LIMPEZA CIRÚRGICA (A "Blindagem") ---
        # 1. Remove blocos de código markdown (```json ... ```)
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```', '', content)
        
        # 2. Tenta encontrar onde começa o JSON '{' e onde termina '}'
        start = content.find('{')
        end = content.rfind('}') + 1
        
        if start != -1 and end != -1:
            clean_json = content[start:end]
            # Faz o parse manual
            data = json.loads(clean_json)
            return SearchIntent(**data)
        else:
            raise ValueError("Nenhum JSON encontrado na resposta.")

    except Exception as e:
        print(f"❌ ERRO NO PARSER DE INTENÇÃO: {e}")
        print(f"   Conteúdo que falhou: {content[:100]}...")
        # Fallback: Busca Global
        return SearchIntent(is_global_query=True)

def build_dynamic_rag_chain(intent: SearchIntent):
    """
    Constrói a pipeline RAG.
    CORREÇÃO CRÍTICA: Desativa o BM25 se houver filtros de entidade para evitar vazamento de contexto.
    """
    global vectorstore_global, keyword_retriever_global, llm_global
    global reranker_global, qa_prompt_global, contextualize_prompt_global

    print(f"\n🔧 Construindo Chain para: Entities={intent.entities} | Topics={intent.topics}")

    # 1. Configurar Filtros do VectorStore (Chroma)
    # 1. Configurar Filtros Dinâmicos (Chroma)
    # Aumentamos K para garantir que pegamos o documento INTEIRO se possível (especialmente se for curto)
    search_kwargs = {"k": 20, "fetch_k": 80}  
    
    filters_list = []
    # has_strict_filter indica se estamos filtrando por UMA entidade específica (ex: Só Fej)
    # Se sim, isso nos permite IGNORAR o BM25 para evitar ruído.
    has_strict_filter = False 

    # --- NOVO: Lógica de Filtro Aprimorada ---
    
    # A. Priorização de Índice (Listagens) - PRIORIDADE MÁXIMA
    # Se requer índice, filtramos por índice INDEPENDENTE se é query global ou não.
    if intent.requires_index:
        print("📑 Detectada intenção de ÍNDICE/LISTAGEM.")
        filters_list.append({"is_index": True})
        if intent.context_filter:
            filters_list.append({"context_type": intent.context_filter})
        
        # Sobrescreve filtro para garantir que pegamos SÓ o índice
        if len(filters_list) > 1:
            search_kwargs["filter"] = {"$and": filters_list} # Tenta ser restritivo
        else:
             search_kwargs["filter"] = filters_list[0]
        
        print(f"🔒 Filtro Chroma (Index): {search_kwargs.get('filter')}")


    # B. Filtros de Entidade/Tópico (Apenas se NÃO for busca global E não for índice já tratado)
    elif not intent.is_global_query:
        
        # B. Filtro Estrito de Entidade (Evita alucinação entre raças)
        # MAS: Se tiver Tópicos junto (Ex: "Fej na Ruptura"), é Cross-Reference. NÃO filtrar estrito se tiver tópico válido.
        if intent.entities and not intent.topics:
            print(f"🎯 Entidade Detectada (Foco Único): {intent.entities[0]}")
            normalized_entity = normalize_topic_name(intent.entities[0])
            
            # CRITICAL FIX: Só aplica filtro estrito se a entidade for um ARQUIVO existente.
            # Caso contrário (ex: "Pedra Viva" que é uma habilidade dentro de um arquivo), 
            # não filtramos metadata, deixamos o vector search achar no conteúdo.
            if normalized_entity in VALID_TOPICS:
                primary_entity = intent.entities[0].strip().title()
                filters_list.append({"entity": primary_entity})
                has_strict_filter = True
            else:
                print(f"⚠️ Entidade '{intent.entities[0]}' não é um arquivo/tópico válido. Entrando em modo 'Busca de Conteúdo'.")
                # Não aplicamos filtro 'entity', o retrieval vai buscar no texto full.

        # C. Validação de Tópicos e Cross-Reference
        elif intent.topics:
            # Separa tópicos válidos (existem no disco) e inválidos
            valid_topics_found = []
            invalid_topics_found = []

            for topic in intent.topics:
                normalized = normalize_topic_name(topic)
                if normalized in VALID_TOPICS:
                    valid_topics_found.append(topic) # Guarda o original para filtro
                else:
                    invalid_topics_found.append(topic)
            
            # Se tem tópicos válidos, aplica filtro estrito DESTE tópico
            if valid_topics_found:
                for topic in valid_topics_found:
                    clean = topic.strip().title()
                    # Tenta ser flexível: Source ou Category
                    filters_list.append({"source": clean})
                    filters_list.append({"category": clean})
                print(f"✅ Tópicos Válidos Filtrados: {valid_topics_found}")
            
            # Se tem tópicos INVÁLIDOS...
            if invalid_topics_found:
                print(f"⚠️ Tópicos Inválidos (Sem arquivo correspondente): {invalid_topics_found}")
                
                # Se TEM entidade E tópico inválido -> INJEÇÃO DE "SEM RELAÇÃO"
                if intent.entities:
                     # NÃO filtramos pelo tópico inválido (para não zerar busca).
                     # Mas avisamos o LLM para checar a relação.
                     print(f"💉 Injetando contexto de 'Verificar Relação' para: {intent.entities} + {invalid_topics_found}")
                     
                     # Adicionamos um filtro de entidade para garantir que achamos algo sobre a entidade pelo menos
                     primary_entity = intent.entities[0].strip().title()
                     filters_list.append({"entity": primary_entity})
                     
                     # A mágica acontece no Prompt do LLM, mas aqui garantimos que o retriever traga dados da entidade
                     # para o LLM poder dizer "Isso é sobre Fej, mas não achei nada sobre Terra do Nunca aqui."
                else:
                    # Se SÓ tem tópico inválido e nenhuma entidade... fallback para busca genérica total (sem filtro)
                    print("⚠️ Apenas tópico inválido detectado. Fallback para busca semântica aberta.")
                    pass

        # Monta o filtro final do Chroma
        if filters_list:
            if len(filters_list) > 1:
                # Se tem vários critérios, usa OR 
                if intent.requires_index:
                     search_kwargs["filter"] = {"is_index": True}
                else:
                     search_kwargs["filter"] = {"$or": filters_list}
            else:
                search_kwargs["filter"] = filters_list[0]
            
            print(f"🔒 Filtro Chroma: {search_kwargs.get('filter')}")

    # 2. Criar Retriever Vetorial
    vector_retriever = vectorstore_global.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    # 3. Seleção de Estratégia de Busca
    # "Faça com que o código sempre utilize a melhor maneira, se precisar ignorar o BM25, pode fazer."
    
    if has_strict_filter:
        print("🚫 ESTRATÉGIA: VECTOR ONLY (Strict). BM25 Desativado para evitar poluição de contexto.")
        base_retriever = vector_retriever
    elif intent.requires_index:
        print("📑 ESTRATÉGIA: VECTOR ONLY (Index Focus). Focando em metadados de índice.")
        base_retriever = vector_retriever
    else:
        # Busca aberta/temática: BM25 ajuda a achar termos exatos no meio de textos grandes
        if keyword_retriever_global:
            print("✅ ESTRATÉGIA: HYBRID (Vector + BM25). Melhor para buscas temáticas ou globais.")
            base_retriever = EnsembleRetriever(
                retrievers=[keyword_retriever_global, vector_retriever],
                weights=[0.4, 0.6]
            )
        else:
            base_retriever = vector_retriever

    # 4. MultiQuery (Opcional - Pode comentar se quiser mais velocidade)
    # Às vezes o MultiQuery também alucina termos. Vamos manter mas com atenção.
    
    # OTIMIZAÇÃO: Se for busca de ÍNDICE, não queremos variações. Queremos o índice.
    if intent.requires_index:
        print("⏩ Pulo MultiQuery para busca de Índice (Foco na exatidão).")
        multi_query_retriever = base_retriever 
    else:
        mq_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "Você é um assistente de busca. Reescreva a pergunta em 3 variações simples para encontrar a resposta no banco de dados."
            "Você está PROIBIDO de tentar achar novos contextos, citando ferramentas ou documentos ou franquias que você não recebeu."),
            ("human", "{question}")
        ])
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm_global,
            prompt=mq_prompt
        )

    # 5. Reranker (Flashrank)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker_global, 
        base_retriever=multi_query_retriever 
    )

    # 6. History Aware
    history_aware_retriever = create_history_aware_retriever(
        llm_global, compression_retriever, contextualize_prompt_global
    )

    # 7. Chain Final
    
    # 7. Chain Final
    # Lógica de Prompt Dinâmico (Injeção de Aviso)
    final_qa_prompt = qa_prompt_global
    
    # Se detectamos necessidade de aviso na validação de tópicos...
    # (Precisamos passar essa info da validação para cá, vamos usar uma variavel local captured)
    # Re-executando a verificação localmente pois o 'filters_list' já foi processado
    
    # Maneira mais limpa: Vamos capturar a mensagem de injeção durante a validação
    injection_msg = ""
    if not intent.is_global_query and intent.topics:
        # Re-check simples ou podemos ter salvo numa var local acima.
        # Vamos confiar que se 'filters_list' tem 'entity' mas não tem o tópico (pq era inválido),
        # podemos inferir ou melhor: fazer a lógica de "warning" ser explícita.
        
        # Recalculando rapidinho para ter certeza (overhead desprezível)
        invalid_topics_found = [t for t in intent.topics if normalize_topic_name(t) not in VALID_TOPICS]
        
        if invalid_topics_found and intent.entities:
             injection_msg = (
                 f"\n\nATENÇÃO DO SISTEMA: O usuário mencionou o tópico '{invalid_topics_found[0]}' que NÃO consta na base de dados. "
                 f"Se você encontrar informações sobre '{intent.entities[0]}', mas nada que o ligue a '{invalid_topics_found[0]}', "
                 f"AVISE O USUÁRIO explicitamente: 'Encontrei informações sobre {intent.entities[0]}, mas não há registros relacionando-o com {invalid_topics_found[0]}'."
             )

    if injection_msg:
        print(f"💉 Criando prompt customizado com aviso: {injection_msg}")
        # Cria um novo prompt template baseada no global + aviso
        system_msg = (
             "Você é um assistente de Base de Conhecimento Especializado. Use o CONTEXTO abaixo para responder."
             "Responda SEMPRE em Português."
             "Você é um assistente estrito de Base de Conhecimento. Use EXCLUSIVAMENTE o contexto fornecido."
             "PROIBIDO usar conhecimentos externos ou de outras franquias"
             "Se o contexto tiver tags [DOC: X], respeite a fonte."
             "Se não souber, diga 'Não consta nos documentos'."
             "NÃO utilize seu conhecimento prévio sobre jogos, filmes ou livros."
             f"{injection_msg}"
             "\n\nCONTEXTO:\n{context}"
        )
        final_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    question_answer_chain = create_stuff_documents_chain(llm_global, final_qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
# --- Lifespan (Inicialização dos Componentes) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore_global, keyword_retriever_global, llm_global
    global reranker_global, qa_prompt_global, contextualize_prompt_global


    print("\n🚀 Inicializando Componentes RAG (Modo Dinâmico)...")

    # 0. Carrega Tópicos Válidos
    load_valid_topics()

    if not os.path.exists(VECTOR_DB_FOLDER):
        print(f"❌ Erro: Pasta '{VECTOR_DB_FOLDER}' não encontrada.")
        yield
        return

    # 1. Componentes Pesados (Carregados 1 vez na memória)
    print("🔹 Carregando Embeddings e ChromaDB...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore_global = Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embedding_function)

    # 2. BM25
    if os.path.exists(BM25_INDEX_FILE):
        print("💾 Carregando BM25...")
        with open(BM25_INDEX_FILE, "rb") as f:
            keyword_retriever_global = pickle.load(f)
    else:
        # Se não tiver BM25, criamos na hora (simplified fallback)
        print("⚠️ Aviso: BM25 index não encontrado. A busca será apenas vetorial neste boot.")
        keyword_retriever_global = None

    # 3. LLM e Reranker
    print("🔹 Inicializando LLM e Reranker...")
    llm_global = OllamaLLM(model=LLM_MODEL, temperature=0.0, num_ctx=16384)
    
    flashrank_client = Ranker(model_name=RERANK_MODEL_NAME, cache_dir="flashrank_cache")
    # Relaxamos o threshold para 0.01 para não descartar informação útil, apenas reordenar.
    reranker_global = FlashrankRerank(client=flashrank_client, top_n=10, score_threshold=0.01)

    # 4. Definição dos Prompts (Fixos)
    contextualize_prompt_global = ChatPromptTemplate.from_messages([
        ("system", (
            "Reformule a pergunta do usuário para ser autossuficiente."
            "Ignore respostas anteriores de 'não sei' ou erros."
            "Retorne APENAS a pergunta reformulada em Português."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_prompt_global = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é um assistente de Base de Conhecimento Especializado. Use o CONTEXTO abaixo para responder."
            "Responda SEMPRE em Português."
            "Você é um assistente estrito de Base de Conhecimento. Use EXCLUSIVAMENTE o contexto fornecido."
            "PROIBIDO usar conhecimentos externos ou de outras franquias. NÃO invente informações."
            "Se o contexto estiver vazio ou não contiver a resposta, diga EXATAMENTE e APENAS: 'Não consta nos documentos'."
            "Se o contexto tiver tags [DOC: X], respeite a fonte."
            "NÃO utilize seu conhecimento prévio sobre jogos, filmes ou livros."
            "\n\nCONTEXTO:\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    print("✅ Sistema Pronto! Chains serão montadas sob demanda.")
    yield
    print("🛑 Desligando...")

app = FastAPI(title="Dynamic RPG RAG API", lifespan=lifespan)

# --- Funções de Stream ---

async def generate_stream(query: str, chat_history: List, model: str, dynamic_chain):
    """
    Função geradora que usa a chain dinâmica criada para este request.
    """
    request_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    sources_text = ""
    try:
        # Usa a chain passada como argumento
        async for chunk in dynamic_chain.astream({"input": query, "chat_history": chat_history}):
            if 'answer' in chunk and chunk['answer']:
                yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk['answer']}, 'finish_reason': None}]})}\n\n"
            
            if 'context' in chunk:
                for doc in chunk['context']:
                    src = doc.metadata.get('source', 'unknown')
                    # Tenta pegar página ou seção se existir
                    loc = doc.metadata.get('Header 1') or doc.metadata.get('page', '?')
                    entry = f"{src} ({loc})"
                    if entry not in sources_text: 
                        sources_text += f"\n- {entry}"
                        
    except Exception as e:
        error_msg = f"[Erro no processamento: {str(e)}]"
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': error_msg}, 'finish_reason': None}]})}\n\n"

    if sources_text:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'\n\n**Fontes:**{sources_text}'}, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

# --- Endpoint Principal ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not vectorstore_global:
        raise HTTPException(status_code=503, detail="Sistema iniciando...")

    raw_messages = request.messages
    query = raw_messages[-1].content
    
    # 1. Tratamento de Histórico (Remove erros anteriores)
    relevant_history = raw_messages[:-1][-MAX_HISTORY_MESSAGES:]
    chat_history = []
    
    for msg in relevant_history:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            # Filtro de Toxicidade de Contexto
            content_lower = msg.content.lower()
            if any(x in content_lower for x in ["não consta", "desculpe", "não encontrei"]):
                continue 
            
            content_clean = msg.content.split("**Fontes:**")[0].strip()
            chat_history.append(AIMessage(content=content_clean))

    # 2. Detecção de Intenção (Router)
    # Descobre se precisa filtrar por "Fej", "História", etc.
    print(f"🤔 Analisando intenção para: '{query}'")
    search_intent = await extract_search_intent(query, llm_global)
    
    # 3. Montagem da Chain Dinâmica
    current_chain = build_dynamic_rag_chain(search_intent)

    # 4. Execução (Stream ou Invoke)
    if request.stream:
        return StreamingResponse(
            generate_stream(query, chat_history, request.model, current_chain), 
            media_type="text/event-stream"
        )

    # Execução normal (Non-stream)
    response = await current_chain.ainvoke({
        "input": query,
        "chat_history": chat_history
    })
    
    answer_content = response['answer']
    if 'context' in response:
        sources = set([f"{doc.metadata.get('source', 'Doc')} ({doc.metadata.get('Header 1', '')})" for doc in response['context']])
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