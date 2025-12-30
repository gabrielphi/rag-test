# Cloud Gemini RAG API

Este projeto implementa uma API de RAG (Retrieval-Augmented Generation) utilizando o modelo Gemini 2.5 Flash do Google e ChromaDB para armazenamento vetorial.

## Visão Geral

A aplicação permite realizar chat com documentos (chat with docs) através de uma API compatível com o formato de mensagens do OpenAI/LibreChat. Ela utiliza:
- **FastAPI**: Para servir a API.
- **LangChain**: Para orquestração do RAG.
- **Google Gemini**: Como LLM (gemini-2.5-flash).
- **ChromaDB**: Como banco de dados vetorial local.
- **HuggingFace Embeddings**: Para gerar embeddings dos documentos.

## Pré-requisitos

- Python 3.12 (Recomendado)
- Chave de API do Google AI Studio (Google API Key)

## Instalação

1. Clone o repositório e navegue até a pasta do projeto:
   ```bash
   cd Cloud-Gemini-LLM
   ```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Configuração

1. Crie um arquivo `.env` na raiz do projeto (use `.env.example` como base se existir) e adicione sua chave de API:
   ```env
   GOOGLE_API_KEY=sua_chave_aqui
   ```

2. Certifique-se de que a pasta `vector_db` existe e está populada. Caso contrário, execute o script de ingestão (se disponível, ex: `ingest.py`):
   ```bash
   python ingest.py
   ```

## Execução

Para iniciar o servidor API no Windows, utilize o seguinte comando:

```powershell
py -3.12 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

A API estará disponível em `http://localhost:8000`.

## Endpoints

### Chat Completions
- **URL**: `/v1/chat/completions`
- **Método**: `POST`
- **Corpo da Requisição**:
  ```json
  {
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Sua pergunta aqui"}
    ],
    "stream": true
  }
  ```
