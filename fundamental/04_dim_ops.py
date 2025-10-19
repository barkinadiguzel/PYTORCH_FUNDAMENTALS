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
