import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
column_names = ["Class Name", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]
data = pd.read_csv(url, names=column_names)

# Drop rows where Class Name is 'B'
data = data[data['Class Name'] != 'B']

# Encode the target variable: 'L' -> 0, 'R' -> 1
data['Class Name'] = data['Class Name'].map({'L': 0, 'R': 1})

# Separate features and target
X = data.drop(columns='Class Name')
y = data['Class Name']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def _init_(self, input_dim):
        super(LogisticRegressionModel, self)._init_()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Initialize the model
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train)
    y_pred_train_class = (y_pred_train >= 0.5).float()
    train_accuracy = accuracy_score(y_train, y_pred_train_class)

    y_pred_test = model(X_test)
    y_pred_test_class = (y_pred_test >= 0.5).float()
    test_accuracy = accuracy_score(y_test, y_pred_test_class)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Testing Accuracy: {test_accuracy:.4f}')