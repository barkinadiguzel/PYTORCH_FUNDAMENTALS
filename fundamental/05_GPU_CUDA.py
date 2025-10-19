import torch

print(torch.rand(3,3))         # Different each time
torch.manual_seed(42)          # generally we use this beacuse we want see similar results with people
print(torch.rand(2,6))         # Always same when seed fixed


print(torch.cuda.is_available())         # Check if CUDA GPU is available
print(torch.cuda.get_device_name(0))     # Get GPU name

print(torch.rand(3,3).cuda())            # Create tensor on GPU

device = "cuda" if torch.cuda.is_available() else "cpu"  
print(device)  

print(torch.cuda.device_count())         # Number of GPUs available

tensor_gpu = tensor4.to(device)          # Move tensor to GPU
print(tensor4.device)                    # Original tensor (CPU)
print(tensor_gpu.device)                 # Now on GPU

tensor_back_to_cpu = tensor_gpu.cpu()    # Move back to CPU
print(tensor_back_to_cpu.device)
