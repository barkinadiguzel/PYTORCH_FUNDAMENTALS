import torch

tensor_dtype = torch.tensor(
    [9,8,7,9],
    dtype=torch.float32,  # Data type (actually float32 is default for pytorch!)
    device="cpu",         # Device (CPU or GPU) 
    requires_grad=True    # Track gradients 
) 
b = torch.tensor([4,5,6], dtype=torch.float32)

print("a + b:", a+b)           # element-wise sum
print("a - b:", a-b)           # element-wise subtraction
print("a * b:", a*b)           # element-wise multiply
print("Dot product (a @ b):", torch.matmul(a,b)) # a.b = 1*4+2*5+3*6 = 32
print("Sum of a:", a.sum())     # collect all the elements
print("Mean of a:", a.mean())   # mean all the elements

print(tensor_dtype + 100)             # Addition (broadcasted)
print(torch.add(tensor_dtype, 5))     # torch.add

print(torch.sub(tensor_dtype, 2))     # Subtraction
print(torch.mul(tensor_dtype, 10))    # Multiplication
print(torch.div(tensor_dtype, 2))     # Division

tensor = torch.tensor([2,4,6])  
print(tensor * tensor)                # Element-wise multiplication

print(torch.matmul(tensor, tensor))   # Matrix multiplication
print(torch.mm(torch.rand(2,3), torch.rand(3,2)))  # Same as matmul but shorter
