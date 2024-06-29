import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Assuming dataset is in a CSV file
df = pd.read_csv('housing_prices.csv')

# Features and target
X = df.drop(columns=['price']).values
y = df['price'].values

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Creating DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class HousingPriceNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingPriceNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


input_dim = X_train.shape[1]
model = HousingPriceNN(input_dim)
 # Loss function
criterion = nn.MSELoss()

# Optimizers
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
    return model



def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())

    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()

    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f'MSE: {mse:.4f}')
    print(f'R-squared: {r2:.4f}')

    # Visualization
    plt.scatter(actuals, predictions, alpha=0.3)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.show()


# Training and evaluation with Adam optimizer
print("Training with Adam optimizer")
model_adam = train_model(model, train_loader, criterion, optimizer_adam)
evaluate_model(model_adam, test_loader)

# Training and evaluation with SGD optimizer
print("Training with SGD optimizer")
model_sgd = train_model(model, train_loader, criterion, optimizer_sgd)
evaluate_model(model_sgd, test_loader)




