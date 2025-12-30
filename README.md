# RAG Test Suite

Este repositório contém uma implementação híbrida de sistemas RAG (Retrieval-Augmented Generation), integrando uma solução local (Self-Hosted) e uma solução em nuvem (Cloud Gemini), ambas acessíveis através de uma interface de chat unificada (LibreChat).

## 🏗️ Arquitetura

O sistema é composto por três partes principais:

1.  **LibreChat (Docker)**: Interface de chat moderna que orquestra as conversas e se comunica com as APIs de RAG.
2.  **Cloud-Gemini-LLM (Docker)**: API RAG que utiliza o Google Gemini 2.5 Flash e ChromaDB. Roda dentro da rede Docker.
3.  **Self-Hosted-LLM (Local Host)**: API RAG que roda diretamente na máquina host, utilizando modelos locais (ex: Llama 3.1 com Ollama).

## 📋 Pré-requisitos

- **Docker Desktop** instalado e rodando.
- **Python 3.12** instalado.
- **Chave de API do Google AI Studio** (para o Gemini).
- **Ollama** (para o Self-Hosted LLM) rodando localmente (opcional, mas necessário para a parte local funcionar plenamente).

## ⚙️ Configuração

### 1. Cloud-Gemini-LLM
Configure a chave de API na pasta `Cloud-Gemini-LLM`:
1.  Entre na pasta: `cd Cloud-Gemini-LLM`
2.  Crie um arquivo `.env` com sua chave:
    ```env
    GOOGLE_API_KEY=sua_chave_aqui
    ```

### 2. Self-Hosted-LLM
Prepare o ambiente local:
1.  Entre na pasta: `cd Self-Hosted-LLM`
2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### 3. LibreChat
A configuração do LibreChat já está definida em `librechat.yaml` e `docker-compose.yml` para conectar aos dois serviços.
- O serviço `Cloud-Gemini-LLM` é acessado via nome de container: `http://cloud-gemini-rag:8001`.
- O serviço `Self-Hosted-LLM` é acessado via gateway do Docker: `http://host.docker.internal:8000`.

## 🚀 Execução

Para rodar todo o sistema, você precisará de **dois terminais**.

### Terminal 1: API Local (Self-Hosted)
Inicie a API que roda fora do Docker:

```powershell
cd Self-Hosted-LLM
py -3.12 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Terminal 2: Docker (LibreChat + Cloud API)
Inicie os serviços Docker na raiz do projeto:

```powershell
docker-compose up
```

## 🌐 Acesso

Abra seu navegador e acesse o LibreChat:

**http://localhost:3080**

Lá você poderá escolher entre os endpoints "Self Hosted RAG" e "Gemini Cloud RAG".

## 🛠️ Troubleshooting

- **Erro de Conexão com Self-Hosted**: Se o LibreChat não conseguir conectar ao `Self-Hosted RAG`, verifique se o Docker consegue resolver `host.docker.internal`. No Windows com WSL2, isso geralmente funciona por padrão.
- **Banco Vetorial Vazio**: Se as respostas forem genéricas, certifique-se de ter rodado os scripts de ingestão (`ingest.py`) dentro de cada pasta de projeto (`Cloud-Gemini-LLM` e `Self-Hosted-LLM`) para popular os bancos de dados vetoriais.
