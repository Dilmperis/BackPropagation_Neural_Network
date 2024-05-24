import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNNpytorch(nn.Module):
    def __init__(self):
        super(SimpleNNpytorch, self).__init__()

        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc1.weight = nn.Parameter(torch.tensor([[1, 1], [1, -1], [-1, 1]], dtype=torch.float64))
        self.fc1.bias = nn.Parameter(torch.tensor([0, 1, -1], dtype=torch.float64))

        self.fc2 = nn.Linear(3, 3, bias=True)
        self.fc2.weight = nn.Parameter(torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, 1]], dtype=torch.float64))
        self.fc2.bias = nn.Parameter(torch.tensor([0, -1, 1], dtype=torch.float64))

        self.fc3 = nn.Linear(3, 2, bias=True)
        self.fc3.weight = nn.Parameter(torch.tensor([[1, 1, 1], [1, -1, -1]], dtype=torch.float64))
        self.fc3.bias = nn.Parameter(torch.tensor([0, 1], dtype=torch.float64))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return x


# Execution
nn_model = SimpleNNpytorch()

# input and target data
x = torch.tensor([0.5, 0.5], dtype=torch.float64)
y = torch.tensor([2, 1], dtype=torch.float64)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(nn_model.parameters(), lr=0.01)
output = nn_model(x)
print(f"Output: {output.detach().numpy()}")

loss = loss_fn(output, y)

# backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")

# Print updated parameters
print("\nUpdated parameters after backpropagation:")
for name, param in nn_model.named_parameters():
    print(f"{name}:")
    print(param.detach().numpy())

