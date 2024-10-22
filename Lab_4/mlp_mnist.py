import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# 2. Model Construction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# Create model and move it to MPS device
model = MLP().to(device)

# 3. Model Compilation
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Model Training
epochs = 20
train_losses = []
valid_losses = []

start_training_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        # Move data and target to MPS device
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_losses.append(epoch_loss / len(train_loader))
    train_accuracy = 100. * correct / len(train_loader.dataset)

    epoch_end_time = time.time()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, '
          f'Accuracy: {train_accuracy:.2f}%, Time: {epoch_end_time - epoch_start_time:.2f}s')

end_training_time = time.time()
print(f'Total training time: {end_training_time - start_training_time:.2f}s')

# 5. Model Evaluation
model.eval()
test_loss = 0
correct = 0

start_evaluation_time = time.time()

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100. * correct / len(test_loader.dataset)

end_evaluation_time = time.time()
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
print(f'Total evaluation time: {end_evaluation_time - start_evaluation_time:.2f}s')

# Optional: Plot training loss
plt.plot(train_losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()