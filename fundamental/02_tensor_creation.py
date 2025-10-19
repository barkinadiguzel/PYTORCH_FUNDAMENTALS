import torch

print("===== Random Tensor =====")
# 3x4 shape, random values between 0 and 1 (rand)
random_tensor = torch.rand(3,4)
print(random_tensor)
print("Shape:", random_tensor.shape)

print("\n===== Zeros Tensor =====")
zeros = torch.zeros(3,4)  # 3x4 shape, all values 0
print(zeros)

print("\n===== Ones Tensor =====")
ones = torch.ones(3,4)   # 3x4 shape, all values 1
print(ones)

print("\n===== Range Tensor =====")
range_tensor = torch.arange(0,10)  # Number tensor from 0 to 9
print(range_tensor)

print("\n===== Zeros like another tensor =====")
zeros_like = torch.zeros_like(range_tensor)  # same shape with range_tensor but fill with 0
print(zeros_like)
