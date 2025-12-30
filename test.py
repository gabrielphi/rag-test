import requests
import json
import sys

# --- Configurações ---
# O endereço onde sua API FastAPI está rodando
API_URL = "http://localhost:8000/v1/chat/completions"
# O nome do modelo deve ser o mesmo que você configurou no backend
MODEL_NAME = "llama3.1:8b" 

def chat_sem_stream(pergunta):
    """
    Modo Clássico: Envia a pergunta, o servidor pensa, e devolve tudo de uma vez.
    Ideal para scripts de automação onde você não tem um usuário esperando na tela.
    """
    print(f"\n🤖 [Sem Stream] Perguntando: '{pergunta}'...")
    print("⏳ Aguardando resposta completa (pode demorar)...")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": pergunta}
        ],
        "stream": False # Desativa o efeito de "digitação"
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Avisa se der erro 400 ou 500
        
        # Pega o JSON e extrai a mensagem
        dados = response.json()
        resposta_texto = dados['choices'][0]['message']['content']
        
        print("-" * 50)
        print(resposta_texto)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Erro: {e}")

def chat_com_stream(pergunta):
    """
    Modo Streaming: A resposta chega pedacinho por pedacinho.
    Ideal para Chatbots, pois o usuário vê que algo está acontecendo.
    """
    print(f"\n🤖 [Com Stream] Perguntando: '{pergunta}'...")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": pergunta}
        ],
        "stream": True # Ativa o modo streaming
    }

    try:
        # stream=True mantém a conexão aberta recebendo dados aos poucos
        with requests.post(API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            
            print("-" * 50)
            # Iteramos linha por linha do que o servidor manda
            for line in response.iter_lines():
                if line:
                    # O servidor manda bytes, precisamos decodificar para texto
                    decoded_line = line.decode('utf-8')
                    
                    # O protocolo SSE sempre começa com "data: "
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line.replace("data: ", "")
                        
                        # Se for o sinal de fim, paramos
                        if json_str.strip() == "[DONE]":
                            break
                        
                        try:
                            # Converte o texto em dicionário Python
                            chunk = json.loads(json_str)
                            
                            # Extrai o pedacinho de texto (se existir)
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', "")
                            
                            if content:
                                # Imprime sem pular linha (end="") e força a saída (flush=True)
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                
                        except json.JSONDecodeError:
                            continue
            print("\n" + "-" * 50)

    except Exception as e:
        print(f"\n❌ Erro de conexão: {e}")

# --- Execução Principal ---
if __name__ == "__main__":
    # Teste 1: Modo Rápido (Streaming)
    chat_com_stream("O que dizem os documentos sobre o tema X?")
    
    # Teste 2: Modo Bloco (Se quiser testar, descomente abaixo)
    # chat_sem_stream("Faça um resumo dos documentos.")