import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer

    def forward(self, x):
        out = self.fc1(x)  # Forward pass through the first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Forward pass through the second layer
        return out


# Parameters
input_size = 784  # Example for MNIST dataset
hidden_size = 128
output_size = 10  # Number of classes in MNIST
learning_rate = 0.001
batch_size = 64
num_epochs = 20

# Initialize the network
model = SimpleNN(input_size, hidden_size, output_size)

# Print the architecture
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Placeholder for loss values
loss_values = []

# Dummy data for input tensor (random example)
# In practice, you will use DataLoader to load your dataset
dummy_input = torch.randn(batch_size, input_size)

# Forward pass example
dummy_output = model(dummy_input)
print("Output:", dummy_output)

# Training loop
for epoch in range(num_epochs):
    # Dummy input and target (random example)
    inputs = torch.randn(batch_size, input_size)
    targets = torch.randint(0, output_size, (batch_size,))

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss curve
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

class EarlyStopping:
    def _init_(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def _call_(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# Parameters for early stopping
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# Modified training loop with early stopping
for epoch in range(num_epochs):
    # Dummy input and target (random example)
    inputs = torch.randn(batch_size, input_size)
    targets = torch.randint(0, output_size, (batch_size,))

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Early stopping check
    if early_stopping(loss.item()):
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Plot the loss curve
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()