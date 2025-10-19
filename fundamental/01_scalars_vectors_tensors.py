import torch

print("===== Scalar =====")
scalar = torch.tensor(8)
print("Scalar:", scalar)           
print("Dimensions (ndim):", scalar.ndim)  # You can see the dimensions

print("\n===== Vector =====")
vector = torch.tensor([7, 7])
print("Vector:", vector)           # You can see the list from there 
print("Shape:", vector.shape)      # vector.shape means you can see their shape 
print("Dimensions:", vector.ndim)  

print("\n===== Matrix =====")
matrix = torch.tensor([[7,8],[4,6]])
print("Matrix:\n", matrix)        # 2x2 matris
print("Shape:", matrix.shape)     # (2,2)
print("Dimensions:", matrix.ndim) # 2 dimension

print("\n===== 3D Tensor =====")
tensor3d = torch.tensor([[[1,2,3],[6,5,2],[7,2,4]]])
print("3D Tensor:\n", tensor3d)    
print("Shape:", tensor3d.shape)    # (1,3,3)
print("Dimensions:", tensor3d.ndim) # 3 dimension
