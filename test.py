import torch
import torchvision
print(torch)
print(torchvision)
print(torch.cuda.is_available())
x=torch.randn(5, 3)
print(x)
print(torch.cuda.device)
print(torch.__version__)
print(torchvision.__version__)
print(torch.version)
print(torch.version.cuda) # Corresponding CUDA version
print(torch.backends.cudnn.version()) # Corresponding cuDNN version
print(torch.cuda.get_device_name(0)) # GPU type
