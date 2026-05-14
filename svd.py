# Simple SVD Example using Synthetic Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# Create synthetic dataset
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

# Scale data
scaler = StandardScaler()

scaled = scaler.fit_transform(df)

# Apply SVD
svd = TruncatedSVD(n_components=2)

svd_data = svd.fit_transform(scaled)

# SVD DataFrame
svd_df = pd.DataFrame(
    svd_data,
    columns=['SVD1', 'SVD2']
)

print("\nSVD Data:\n")
print(svd_df.head())

# Explained variance
print(
    "\nExplained Variance Ratio:\n",
    svd.explained_variance_ratio_
)

# Visualization
plt.scatter(
    svd_df['SVD1'],
    svd_df['SVD2']
)

plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")

plt.title("SVD on Synthetic Dataset")

plt.grid(True)
plt.show()