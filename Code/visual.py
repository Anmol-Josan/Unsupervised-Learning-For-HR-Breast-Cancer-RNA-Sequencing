import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

# Load the processed integrated data
adata = sc.read_h5ad('Archive/Processed_Data/processed_s_rna_seq_data_integrated.h5ad')

# Compute t-SNE if not present
if 'X_tsne' not in adata.obsm:
    sc.tl.tsne(adata)

# Load the cluster labels
labels = np.load('Code/agglomerative_3_complete_combined_scaled_labels.npy')

# Get the t-SNE coordinates
# X_pca, X_pca_harmony, X_umap
tsne_coords = adata.obsm['X_tsne']

# Create a color map for the clusters
unique_labels = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

# Plot the clusters
plt.figure(figsize=(12, 10))
for i, label in enumerate(unique_labels):
    mask = labels == label
    plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                color=colors[i], label=f'Cluster {label}', alpha=0.7, s=10)

plt.title('Agglomerative Clustering: 3 Clusters (Complete Linkage, Combined Scaled Features)', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=14)
plt.ylabel('t-SNE Dimension 2', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print cluster sizes
cluster_counts = pd.Series(labels).value_counts().sort_index()
print("Cluster sizes:")
print(cluster_counts)