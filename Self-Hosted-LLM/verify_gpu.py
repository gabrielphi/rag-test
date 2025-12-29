import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Opa! O Python não está vendo sua GPU. Você provavelmente instalou o PyTorch errado.")