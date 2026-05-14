# Simple EM Algorithm Example using GMM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
np.random.seed(42)

size = 50

x = np.concatenate([
    np.random.randint(20,40,size),
    np.random.randint(60,80,size),
    np.random.randint(20,40,size)
])

y = np.concatenate([
    np.random.randint(20,40,size),
    np.random.randint(60,80,size),
    np.random.randint(60,80,size)
])

df = pd.DataFrame({
    'Feature1': x,
    'Feature2': y
})

print(df.head())

# Scale data
scaler = StandardScaler()

scaled = scaler.fit_transform(df)

# Train GMM model
model = GaussianMixture(
    n_components=3,
    random_state=42
)

model.fit(scaled)

# Predict clusters
df['Cluster'] = model.predict(scaled)

# Responsibilities (E-Step probabilities)
print("\nResponsibilities:\n",
      model.predict_proba(scaled)[:5])

# Log Likelihood
print("\nLog Likelihood:",
      model.score(scaled))

# Test prediction
test = pd.DataFrame({
    'Feature1': [25],
    'Feature2': [27]
})

test_scaled = scaler.transform(test)

result = model.predict(test_scaled)

prob = model.predict_proba(test_scaled)

print("\nPredicted Cluster:", result[0])

print("\nCluster Probabilities:\n", prob)

# Visualization
plt.scatter(
    df['Feature1'],
    df['Feature2'],
    c=df['Cluster']
)

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("EM Clustering using GMM")

plt.grid(True)
plt.show()