import torch

print(f"Versão do Python: {torch.__version__}")
print(f"CUDA está disponível? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Versão do CUDA no PyTorch: {torch.version.cuda}")
    
    # Teste de alocação simples
    x = torch.tensor([1.0, 2.0]).to("cuda")
    print("Teste de tensor na GPU: Sucesso!")
else:
    print("Erro: A GPU não foi detectada pelo PyTorch.")