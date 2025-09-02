# =======================
# PyTorch Fundamentals
# =======================

import torch
import numpy as np
import matplotlib.pyplot as plt


# =======================
# Scalars, Vectors, Matrices, Tensors
# =======================

scalar = torch.tensor(8)  
print(scalar)             # A scalar (0-dimensional tensor)
print(scalar.ndim)        # Number of dimensions (0 for scalar)
print(scalar.item())      # Convert tensor to Python number

vector = torch.tensor([7, 7])  
print(vector)             # A vector (1-dimensional tensor)
print(vector.ndim)        # 1 dimension
print(vector.shape)       # Shape = (2,)

matrix = torch.tensor([[7, 8], [4, 6]])  
print(matrix)             # A 2D tensor (matrix)

tensor3d = torch.tensor([[[1,2,3],[6,5,2],[7,2,4]]])  
print(tensor3d.shape)     # Shape = (1, 3, 3)

tensor4d = torch.tensor([[[9,8,7,9], [5,2,1,2]]])  
print(tensor4d)           
print(tensor4d.ndim)      # 3D tensor
print(tensor4d.shape)     # Shape = (1, 2, 4)


# =======================
# Creating Tensors
# =======================

random_tensor = torch.rand(3,4)  
print(random_tensor)      # Random values between 0 and 1

# Typical image tensor → (Channels, Height, Width)
random_image_tensor = torch.rand([3, 224, 224])  
print(random_image_tensor.shape)

zeros = torch.zeros(size=(3,4))  
print(zeros)              # Matrix filled with zeros

ones = torch.ones(size=(3,4))  
print(ones)               # Matrix filled with ones

range_tensor = torch.arange(0,10)  
print(range_tensor)       # Values from 0 to 9

zeros_like = torch.zeros_like(input=range_tensor)  
print(zeros_like)         # Tensor with same shape as range_tensor, filled with zeros


# =======================
# Data Types and Devices
# =======================

tensor_dtype = torch.tensor(
    [9,8,7,9],
    dtype=torch.float32,  # Data type
    device="cpu",         # Device (CPU or GPU)
    requires_grad=True    # Track gradients
)
print(tensor_dtype.dtype)  # float32

float16_tensor = tensor_dtype.type(torch.float16)  
print(float16_tensor)      # Convert dtype to float16
print(float16_tensor.shape)
print(float16_tensor.device)


# =======================
# Basic Math Operations
# =======================

print(tensor_dtype + 100)             # Addition (broadcasted)
print(torch.add(tensor_dtype, 5))     # torch.add

print(torch.sub(tensor_dtype, 2))     # Subtraction
print(torch.mul(tensor_dtype, 10))    # Multiplication
print(torch.div(tensor_dtype, 2))     # Division

tensor = torch.tensor([2,4,6])  
print(tensor * tensor)                # Element-wise multiplication

print(torch.matmul(tensor, tensor))   # Matrix multiplication
print(torch.mm(torch.rand(2,3), torch.rand(3,2)))  # Same as matmul but shorter


# =======================
# Matrix Operations
# =======================

A = torch.tensor([[3,2],[3,7],[7,2]])  
B = torch.tensor([[1,2],[4,7],[7,2]])  

print(torch.mm(A, B.T))   # Matrix multiplication with transpose

x = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])  
print(x, x.shape)

print(torch.min(x))       # Minimum value
print(torch.max(x))       # Maximum value
print(torch.mean(x.type(torch.float32)))  # Mean (requires float type)
print(x.sum())            # Sum of elements
print(x.argmin())         # Index of min value
print(x.argmax())         # Index of max value

x_reshape = x.reshape(2,5)  
print(x_reshape)          # Reshape to 2x5

z = x.view(2,5)           # Another way (shares memory with x)
print(z)

x_stacked = torch.stack([x,x,x,x], dim=0)  
y_stacked = torch.stack([x,x,x,x], dim=1)  
print(x_stacked)          # Stack along new dimension 0
print(y_stacked)          # Stack along new dimension 1


# =======================
# Dimension Operations
# =======================

x_sqzd = x.squeeze()  
print(x_sqzd, x_sqzd.shape)   # Remove dimensions of size 1

x_unsqzd = x.unsqueeze(dim=0)  
print(x_unsqzd.shape)         # Add an extra dimension

print(torch.permute(x, (1,0)).shape)  # Swap dimensions

y = torch.arange(1,10).reshape(1,3,3)  
print(y)  
print(y[:,:,2])               # Select along last dimension


# =======================
# NumPy ↔ PyTorch
# =======================

tensor4 = torch.tensor([1,2,3,4])  

array = np.arange(1,12)  
tensor_from_np = torch.from_numpy(array)  
print(tensor_from_np.dtype)    # Default dtype = int64

tensor1 = torch.ones(7)  
nump_tens = tensor1.numpy()  
print(nump_tens)               # Convert tensor to numpy


# =======================
# Random Seeds
# =======================

print(torch.rand(3,3))         # Different each time
torch.manual_seed(42)  
print(torch.rand(2,6))         # Always same when seed fixed


# =======================
# GPU / CUDA
# =======================

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
