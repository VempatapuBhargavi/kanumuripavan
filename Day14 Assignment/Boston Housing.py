import itertools

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Load the dataset
boston_housing = load_boston()
X, y = boston_housing.data, boston_housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class SimpleLinearRegressionModel(nn.Module):
    def _init_(self, input_dim):
        super(SimpleLinearRegressionModel, self)._init_()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Hyperparameters grid
param_dist = {
    'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
    'epochs': [50, 100, 200, 300]
}


class PyTorchRegressor:
    def _init_(self, model, learning_rate=0.01, epochs=50):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy().flatten()


# Wrapper function for RandomizedSearchCV
def create_model(learning_rate, epochs):
    model = SimpleLinearRegressionModel(input_dim=X_train.shape[1])
    return PyTorchRegressor(model, learning_rate=learning_rate, epochs=epochs)


# Convert the parameter distribution to a list of hyperparameter combinations
import random

param_combinations = [dict(zip(param_dist.keys(), v)) for v in itertools.product(*param_dist.values())]
random.shuffle(param_combinations)

best_model = None
best_params = None
best_score = float('inf')

# Perform random search by sampling a subset of parameter combinations
n_iter = 10  # number of random parameter sets to try
sampled_params = random.sample(param_combinations, n_iter)

for params in sampled_params:
    model = create_model(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    if mse < best_score:
        best_score = mse
        best_params = params
        best_model = model

print(f"Best Parameters: {best_params}")
print(f"Best MSE: {best_score}")
# Calculate and interpret regression metrics for the best model
best_predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, best_predictions)
mse = mean_squared_error(y_test, best_predictions)
rmse = mean_squared_error(y_test, best_predictions, squared=False)
r2 = r2_score(y_test, best_predictions)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")