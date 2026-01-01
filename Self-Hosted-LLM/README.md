# Self-Hosted Generic RAG System

Este projeto é um sistema de **RAG (Retrieval-Augmented Generation)** hospedado localmente, projetado para transformar qualquer coleção de documentos de texto em uma Base de Conhecimento Inteligente.

Diferente de sistemas rígidos, este projeto se adapta à **sua estrutura de pastas**. Não importa se você está organizando documentos jurídicos, técnicos, receitas ou campanhas de RPG: a pasta define o contexto.

## 🚀 Principais Funcionalidades

-   **Modelos Locais (Ollama)**: Privacidade total. Seus documentos nunca saem da sua máquina.
-   **Categorização Dinâmica**: O sistema entende o contexto baseado no nome das suas pastas (Ex: `Marketing/CampanhaQ1.txt` -> Contexto: Marketing, Entidade: CampanhaQ1).
-   **Busca Híbrida Inteligente**: Combina **Vetores** (significado) com **BM25** (palavras-chave).
-   **Índices e Listas**: Prioriza arquivos de índice (ex: `00_Resumo.txt`) quando você pede uma visão geral.
-   **Cross-Reference**: Entende quando você pergunta sobre "Projeto X" no contexto de "Financeiro" e cruza as informações.

## 📂 Como Organizar seus Documentos

A "inteligência" do sistema vem da sua organização. Use a pasta `documentos/` como raiz.

### Estrutura Recomendada

```text
documentos/
├── [CATEGORIA 1] (Ex: Tecnologia)
│   ├── [ENTIDADE A].txt (Ex: Python.txt)
│   ├── [ENTIDADE B].txt (Ex: Docker.txt)
│   └── 00_INDICE_TECNOLOGIA.txt (Resumo geral desta pasta)
│
├── [CATEGORIA 2] (Ex: Recursos Humanos)
│   ├── Politica_Ferias.txt
│   ├── Onboarding.txt
│   └── ...
```

-   **Nível 1 (Pastas)**: Define a **Categoria Geral** (Contexto).
-   **Arquivos**: Cada arquivo é tratado como uma **Entidade** ou Tópico Específico.
-   **Índices**: Arquivos começando com `00_` ou contendo `INDICE` no nome são tratados como prioritários para listagens.

## 🛠️ Instalação e Uso

### Pré-requisitos
-   Python 3.12+
-   [Ollama](https://ollama.ai/) instalado e rodando.
-   Modelo LLM baixado (Recomendado: `gemma2:9b` ou `llama3`).

### 1. Configuração
1.  Renomeie `.env.example` para `.env`.
2.  Edite `.env` e ajuste `LLM_MODEL` se necessário.

### 2. Ingestão de Dados
Sempre que adicionar novos arquivos na pasta `documentos/`, rode:
```bash
py -3.12 ingest.py
```
Isso vai ler, categorizar e criar o "cérebro" vetorial do sistema.

### 3. Rodando o Chat
Para iniciar a API e começar a conversar:
```bash
py -3.12 api.py
```
Acesse a interface de documentação (Swagger) em: `http://localhost:8000/docs`

## 🧠 Exemplos de Uso

-   **Pergunta Específica**: *"O que a politica de férias diz sobre hora extra?"*
    -   O sistema detecta a entidade "Politica de Ferias" e busca exatamente lá.
-   **Pergunta Geral**: *"Quais tecnologias usamos?"*
    -   O sistema busca nos índices da pasta Tecnologia.
-   **Cruzamento**: *"Como o Docker impacta o Onboarding?"*
    -   O sistema busca informações tanto de Tecnologia/Docker quanto de RH/Onboarding.
