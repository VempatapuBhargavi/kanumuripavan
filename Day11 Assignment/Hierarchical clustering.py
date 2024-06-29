
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_df = pd.read_csv(url, delimiter=';')

# Normalize the features
scaler = StandardScaler()
wine_data_scaled = scaler.fit_transform(wine_df)

# Apply hierarchical clustering
Z = linkage(wine_data_scaled, method='ward')

# Create and visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Define the number of clusters
num_clusters = 3

# Assign cluster labels
labels = fcluster(Z, num_clusters, criterion='maxclust')

# Compute the silhouette score
sil_score = silhouette_score(wine_data_scaled, labels)
print(f'Silhouette Score: {sil_score:.3f}')