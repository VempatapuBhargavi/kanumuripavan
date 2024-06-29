import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
# For this example, we will use the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Perform polynomial regression
# Add polynomial features up to degree 3
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predict using the polynomial regression model
y_pred_poly = poly_model.predict(X_test_poly)

# Calculate the Mean Squared Error (MSE) for the polynomial regression model
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression MSE: {mse_poly:.2f}")

# Perform linear regression
# Train the linear regression model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Predict using the linear regression model
y_pred_lin = lin_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) for the linear regression model
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f"Linear Regression MSE: {mse_lin:.2f}")

# Compare the performance of the models
if mse_poly < mse_lin:
    print("Polynomial Regression model performs better.")
else:
    print("Linear Regression model performs better.")