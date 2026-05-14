# Simple LDA Example using Synthetic Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
np.random.seed(42)
size = 100
# Class 0
x1 = np.random.normal(30, 5, size)
y1 = np.random.normal(30, 5, size)
# Class 1
x2 = np.random.normal(70, 5, size)
y2 = np.random.normal(70, 5, size)
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
target = np.concatenate([
    np.zeros(size),
    np.ones(size)
])

df = pd.DataFrame({
    'Feature1': x,
    'Feature2': y,
    'Target': target
})

print(df.head())

# Features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Scale data
scaler = StandardScaler()

scaled = scaler.fit_transform(X)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=1)

lda_data = lda.fit_transform(scaled, y)

# LDA DataFrame
lda_df = pd.DataFrame({
    'LDA1': lda_data.flatten(),
    'Target': y
})

print("\nLDA Data:\n")
print(lda_df.head())

# Accuracy
pred = lda.predict(scaled)

print(
    "\nAccuracy:",
    lda.score(scaled, y)
)

# Visualization
plt.scatter(
    lda_df['LDA1'],
    np.zeros(len(lda_df)),
    c=lda_df['Target']
)

plt.xlabel("LDA Component 1")

plt.title("LDA on Synthetic Dataset")

plt.grid(True)

plt.show()