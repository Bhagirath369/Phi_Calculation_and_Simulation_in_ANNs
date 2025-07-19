import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 1. Setup & Load MNIST

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# 2. Define Small MLP (6 hidden neurons)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 6)   # small hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.fc1(x)
        self.hidden = h.detach()      # save raw hidden activations
        x = self.relu(h)
        x = self.fc2(x)
        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. Model Train
epochs = 5
model.train()
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# 4. After Training: Capture 201 Samples
model.eval()
activation_list = []
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, (img, label) in enumerate(data_loader):
        _ = model(img)
        activation = model.hidden.squeeze().numpy()  # shape: (6,)
        activation_list.append(activation)

        if i >= 200:  # capture 201 samples → 200 transitions
            break

# 5. Binarize the Activations

activations = np.array(activation_list)  # shape: (201, 6)

# Method: Median threshold per neuron
thresholds = np.median(activations, axis=0)
binarized = (activations > thresholds).astype(int)  # shape: (201, 6)


#  Ready for TPM Construction

print("Sample binarized activations (first 5 rows):")
print(binarized[:20])
np.save('binarized_states.npy', binarized)

# 6. Evaluate Accuracy on Test Set
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

