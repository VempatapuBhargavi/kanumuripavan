import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
boston = load_boston()
data = boston.data
targets = boston.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors for linear regression
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def _init_(self, input_dim, output_dim):
        super(LinearRegressionModel, self)._init_()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


input_dim = X_train.shape[1]
output_dim = 1
linear_model = LinearRegressionModel(input_dim, output_dim)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.01)

# Train the linear model
num_epochs = 1000
for epoch in range(num_epochs):
    linear_model.train()

    # Forward pass
    outputs = linear_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Linear Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the linear model
linear_model.eval()
with torch.no_grad():
    train_pred_linear = linear_model(X_train_tensor)
    test_pred_linear = linear_model(X_test_tensor)
    train_mse_linear = mean_squared_error(y_train, train_pred_linear.numpy())
    test_mse_linear = mean_squared_error(y_test, test_pred_linear.numpy())

# Ridge regression with cross-validation
ridge = Ridge()
alphas = np.logspace(-4, 4, 50)
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alphas}, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train, y_train)
best_ridge_alpha = ridge_cv.best_params_['alpha']

# Train Ridge regression with the best alpha
ridge_best = Ridge(alpha=best_ridge_alpha)
ridge_best.fit(X_train, y_train)
train_pred_ridge = ridge_best.predict(X_train)
test_pred_ridge = ridge_best.predict(X_test)
train_mse_ridge = mean_squared_error(y_train, train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, test_pred_ridge)

# Lasso regression with cross-validation
lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(lasso, param_grid={'alpha': alphas}, scoring='neg_mean_squared_error', cv=5)
lasso_cv.fit(X_train, y_train)
best_lasso_alpha = lasso_cv.best_params_['alpha']

# Train Lasso regression with the best alpha
lasso_best = Lasso(alpha=best_lasso_alpha)
lasso_best.fit(X_train, y_train)
train_pred_lasso = lasso_best.predict(X_train)
test_pred_lasso = lasso_best.predict(X_test)
train_mse_lasso = mean_squared_error(y_train, train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, test_pred_lasso)

# Print the results
print(f'Linear Regression - Train MSE: {train_mse_linear:.4f}, Test MSE: {test_mse_linear:.4f}')
print(f'Ridge Regression (alpha={best_ridge_alpha}) - Train MSE: {train_mse_ridge:.4f}, Test MSE: {test_mse_ridge:.4f}')
print(f'Lasso Regression (alpha={best_lasso_alpha}) - Train MSE: {train_mse_lasso:.4f}, Test MSE: {test_mse_lasso:.4f}')

# Plot the predicted vs actual values for the test set
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_test, test_pred_linear, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression')

plt.subplot(1, 3, 2)
plt.scatter(y_test, test_pred_ridge, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Ridge Regression')

plt.subplot(1, 3, 3)
plt.scatter(y_test, test_pred_lasso, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Lasso Regression')

plt.tight_layout()
plt.show()