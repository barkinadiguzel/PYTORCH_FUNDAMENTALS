import torch

a = torch.tensor([1,2,3], dtype=torch.float32) #float32 is default for pytorch!
b = torch.tensor([4,5,6], dtype=torch.float32)

print("a + b:", a+b)           # element-wise sum
print("a - b:", a-b)           # element-wise subtraction
print("a * b:", a*b)           # element-wise multiply
print("Dot product (a @ b):", torch.matmul(a,b)) # a.b = 1*4+2*5+3*6 = 32
print("Sum of a:", a.sum())     # collect all the elements
print("Mean of a:", a.mean())   # mean all the elements
