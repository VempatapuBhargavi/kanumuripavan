import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(url, header=None, names=columns)

# Drop the class column as we only need the features for clustering
iris_data = iris_df.drop(columns=['class'])

# Normalize the features
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)

# Convert to tensor
iris_tensor = torch.tensor(iris_data_scaled, dtype=torch.float)

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def closest_centroid(X, centroids):
    distances = torch.cdist(X, centroids)
    return torch.argmin(distances, dim=1)

def compute_centroids(X, labels, k):
    centroids = torch.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        centroids[i] = points.mean(dim=0)
    return centroids

def kmeans(X, k, num_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(num_iters):
        labels = closest_centroid(X, centroids)
        new_centroids = compute_centroids(X, labels, k)
        if torch.equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Set number of clusters
k = 3

# Run k-Means clustering
centroids, labels = kmeans(iris_tensor, k)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_data_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
for i in range(k):
    points = iris_pca[labels.numpy() == i]
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
plt.title('k-Means Clustering of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Compute the silhouette score
sil_score = silhouette_score(iris_data_scaled, labels.numpy())
print(f'Silhouette Score: {sil_score:.3f}')
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(url, header=None, names=columns)

# Drop the class column as we only need the features for clustering
iris_data = iris_df.drop(columns=['class'])

# Normalize the features
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)

# Convert to tensor
iris_tensor = torch.tensor(iris_data_scaled, dtype=torch.float)

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def closest_centroid(X, centroids):
    distances = torch.cdist(X, centroids)
    return torch.argmin(distances, dim=1)

def compute_centroids(X, labels, k):
    centroids = torch.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        centroids[i] = points.mean(dim=0)
    return centroids

def kmeans(X, k, num_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(num_iters):
        labels = closest_centroid(X, centroids)
        new_centroids = compute_centroids(X, labels, k)
        if torch.equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Set number of clusters
k = 3

# Run k-Means clustering
centroids, labels = kmeans(iris_tensor, k)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_data_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
for i in range(k):
    points = iris_pca[labels.numpy() == i]
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
plt.title('k-Means Clustering of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Compute the silhouette score
sil_score = silhouette_score(iris_data_scaled, labels.numpy())
print(f'Silhouette Score: {sil_score:.3f}')