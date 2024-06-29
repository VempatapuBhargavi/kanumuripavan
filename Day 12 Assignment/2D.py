import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the Wine Quality dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, delimiter=';')

# Step 2: Preprocess the data
features = df.drop('quality', axis=1).values
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Step 3: Apply t-SNE
tsne_2d = TSNE(n_components=2, random_state=42)
tsne_3d = TSNE(n_components=3, random_state=42)

features_2d = tsne_2d.fit_transform(features_normalized)
features_3d = tsne_3d.fit_transform(features_normalized)

# Step 4: Visualize the results

# 2D visualization
plt.figure(figsize=(10, 6))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=df['quality'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Wine Quality')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('t-SNE 2D Projection of Wine Quality Dataset')
plt.show()

# 3D visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=df['quality'], cmap='viridis', alpha=0.7)
plt.colorbar(sc, label='Wine Quality')
ax.set_xlabel('TSNE Component 1')
ax.set_ylabel('TSNE Component 2')
ax.set_zlabel('TSNE Component 3')
plt.title('t-SNE 3D Projection of Wine Quality Dataset')
plt.show()