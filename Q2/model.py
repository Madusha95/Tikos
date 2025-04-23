import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and prepare the data
data = pd.read_csv('mnist_train.csv')
data = np.array(data)
np.random.shuffle(data)

X = data[:, 1:] / 255.0
Y = data[:, 0]

X_train = X[:int(0.8*len(X))]
Y_train = Y[:int(0.8*len(Y))]
X_val = X[int(0.8*len(X)):]
Y_val = Y[int(0.8*len(Y)):]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

# DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the model
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # CrossEntropyLoss applies softmax automatically

# Initialize model, loss, optimizer
model = DigitRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10):  # adjust epochs if needed
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate accuracy on train set every epoch
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_preds = torch.argmax(train_outputs, dim=1)
        train_acc = (train_preds == Y_train_tensor).float().mean()
        print(f"Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}")

# Evaluate on validation set
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_preds = torch.argmax(val_outputs, dim=1)
    val_acc = (val_preds == Y_val_tensor).float().mean()
    print(f"Validation Accuracy: {val_acc:.4f}")

# Show prediction for one sample
sample_index = 560
sample_image = X_val_tensor[sample_index].reshape(28, 28)
sample_output = model(X_val_tensor[sample_index])
predicted_label = torch.argmax(sample_output).item()
print(f"Predicted: {predicted_label}, Actual: {Y_val[sample_index]}")

plt.imshow(sample_image, cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.show()

# Save the model
torch.save(model.state_dict(), 'digit_recognizer_model.pth')
