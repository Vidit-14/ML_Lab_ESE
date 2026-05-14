import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

size = 200

x1 = np.random.normal(50, 10, size)

x2 = x1 * 0.8 + np.random.normal(0, 5, size)

x3 = x1 * 0.5 + x2 * 0.3 + np.random.normal(0, 3, size)

x4 = np.random.normal(30, 15, size)

df = pd.DataFrame({
    'Feature1': x1,
    'Feature2': x2,
    'Feature3': x3,
    'Feature4': x4
})

print(df.head())
scaler = StandardScaler()
scaler = scaler.fit_transform(df)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaler)
# PCA DataFrame
pca_df = pd.DataFrame(
    pca_data,
    columns=['PC1', 'PC2']
)

print("\nPCA Data:\n")
print(pca_df.head())

# Explained Variance
print(
    "\nExplained Variance Ratio:\n",
    pca.explained_variance_ratio_
)

# Visualization
plt.scatter(
    pca_df['PC1'],
    pca_df['PC2']
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA on Synthetic Dataset")

plt.grid(True)
plt.show()

