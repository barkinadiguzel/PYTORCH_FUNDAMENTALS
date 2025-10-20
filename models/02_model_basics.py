import torch
import torch.nn as nn
import torch.nn.functional as F

# basic fully connected model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# sample: input 10 dim, hidden 5, output 2
model = SimpleModel(10, 5, 2)


# dummy data
x = torch.randn(3, 10)  # 3 sample, 10 feature
y = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)

criterion = nn.MSELoss()  # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # optimizer

for epoch in range(5):  # 5 epoch
    optimizer.zero_grad()       # reset the gradient
    outputs = model(x)          # forward pass
    loss = criterion(outputs, y)  # calculate the loss
    loss.backward()             # backward pass
    optimizer.step()            # update the parametres
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Save
torch.save(model.state_dict(), "simple_model.pth")

# Load
loaded_model = SimpleModel(10, 5, 2)
loaded_model.load_state_dict(torch.load("simple_model.pth"))
loaded_model.eval()  # open the test mode


