import torch

print(f"Cuda is available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"GPU names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
