# Its my first model we can take something from here
import torch
from torch import nn  # nn for neural networks
import matplotlib.pyplot as plt

# Linear Regression Model
class LinearRegressionModel(nn.Module): # we generally take parents class as nn.Module 
    def __init__(self):
        super().__init__()
        # we define weight and bias with nn.Parameter 
        # Thus, the optimizer can update these values
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))

    def forward(self, x: torch.Tensor):
        #the formula of linear regression: y = w*x + b we change b and w (b: bias,w: weight)
        return self.weights * x + self.bias

torch.manual_seed(42)
# create the model
model0 = LinearRegressionModel()

# Print the model parameters
print(list(model0.parameters()))
print(model0.state_dict())





