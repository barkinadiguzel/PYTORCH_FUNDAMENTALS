import torch

x = torch.arange(1,13)
print("Original tensor:", x)
print("Shape:", x.shape)       # 1D tensor, 12 element

# Reshape: change the tensor shape, The total number of elements must be the same
x_reshaped = x.reshape(3,4)   
print("Reshaped 3x4:\n", x_reshaped)

# Unsqueeze: add a new dimension
x_unsq = x.unsqueeze(0)
print("Unsqueeze dimension 0:", x_unsq.shape)  # (1,12)

# Squeeze: Remove dimensions with a size of 1
x_sq = x_unsq.squeeze()
print("Squeeze:", x_sq.shape)  # (12,)

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

