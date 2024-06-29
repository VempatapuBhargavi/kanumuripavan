import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the Wine Quality dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, delimiter=';')

# Step 2: Preprocess the data
features = df.drop('quality', axis=1).values
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Step 3: Implement PCA using PyTorch

# Convert the numpy array to a PyTorch tensor
X = torch.tensor(features_normalized, dtype=torch.float32)

# Compute the covariance matrix
X_centered = X - X.mean(dim=0)
cov_matrix = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = torch.argsort(eigenvalues, descending=True)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the first two principal components
PCA_components = eigenvectors[:, :2]
X_pca = torch.matmul(X_centered, PCA_components)

# Step 4: Visualize the results
X_pca_np = X_pca.detach().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_pca_np[:, 0], X_pca_np[:, 1], c=df['quality'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Wine Quality')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Wine Quality Dataset')
plt.show()