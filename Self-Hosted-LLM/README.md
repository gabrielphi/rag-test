# 🧠 Local RAG Chatbot

Uma aplicação de **RAG (Retrieval-Augmented Generation)** totalmente local, que permite conversar com seus documentos PDF e TXT usando **Ollama** e **LangChain**.

## 🚀 Funcionalidades

- **100% Local**: Nenhum dado sai da sua máquina.
- **Suporte a PDFs e TXT**: Ingestão de múltiplos arquivos.
- **Citações**: Indica exatamente qual documento e página foi usado para a resposta.
- **Embeddings Multilíngues**: Configurado com `paraphrase-multilingual-MiniLM-L12-v2` para melhor performance em Português.

## 📋 Pré-requisitos

1. **Python 3.12+** instalado.
2. **[Ollama](https://ollama.com/)** instalado e rodando.
3. Modelo **Llama 3.2** (3B) baixado no Ollama:
   ```bash
   ollama pull llama3.2:3b
   ```

## 🛠️ Instalação

1. Clone ou baixe este repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Como Usar

### 1. Preparar Documentos
Coloque seus arquivos `.pdf` e `.txt` dentro da pasta:
```
/documentos
```

### 2. Criar Banco Vetorial (Ingestão)
Execute o script de ingestão sempre que adicionar novos arquivos. Ele processará os textos e salvará no banco de dados local (`vector_db`).
```bash
python ingest.py
```
*Saída esperada:*
```
✅ [PDF] Carregado: 'Manual Beneficios 2024' (12 páginas)
🧠 Gerando embeddings...
🚀 Sucesso! Banco vetorial salvo em 'vector_db'.
```

### 3. Iniciar o Chat
Execute o aplicativo principal para conversar com seus dados.
```bash
python app.py
```

### 4. Interagindo
- Digite sua pergunta e pressione Enter.
- O sistema buscará os 3 trechos mais relevantes e gerará uma resposta.
- Digite `sair` para encerrar.

## 📂 Estrutura do Projeto

- `app.py`: Script principal do chat (interface usuário).
- `ingest.py`: Script para processar documentos e criar o banco vetorial.
- `requirements.txt`: Lista de dependências Python.
- `documentos/`: Pasta onde você coloca seus arquivos (PDF/TXT).
- `vector_db/`: Pasta gerada automaticamente contendo o banco de dados vetorial (ChromaDB).

## ⚠️ Solução de Problemas comuns

**Erro: `vector_db` não encontrado**
- Rode `python ingest.py` primeiro.

**Erro: `Dimension mismatch`**
- Certifique-se de que `app.py` e `ingest.py` usem o mesmo `EMBEDDING_MODEL_NAME`.
- Se mudou o modelo, delete a pasta `vector_db` e rode `ingest.py` novamente.

**Erro: Ollama connection refused**
- Verifique se o aplicativo do Ollama está aberto e rodando em background.
