import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import*
np.random.seed(42)
size=50
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
    'feature1':x,
    'feature2':y
})

print(df.head())
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
model = KMeans(n_clusters =3, random_state=42)

# Elbow Method

# inertia_values = []

# for k in range(1, 11):

#     model = KMeans(
#         n_clusters=k,
#         random_state=42
#     )

#     model.fit(scaled)

#     inertia_values.append(model.inertia_)

# # Plot Elbow Graph
# plt.plot(
#     range(1, 11),
#     inertia_values,
#     marker='o'
# )

# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.grid(True)
# plt.show()

df['Cluster'] = model.fit_predict(scaled)
print('\nCentroids:\n',scaler.inverse_transform(model.cluster_centers_))

# Evaluation Metrics
print("\nInertia:", model.inertia_)

print(
    "Silhouette Score:",
    silhouette_score(scaled, df['Cluster'])
)

test = pd.DataFrame({
    'feature1': [30],
    'feature2': [40]
})

test_scaled = scaler.fit_transform(test)
pred = model.predict(test_scaled)
print("\nPredicted Cluster:", pred[0])

# Visualization

plt.scatter(
    df['feature1'],
    df['feature2'],
    c=df['Cluster']
)

# Correct centroid plotting
centroids = scaler.inverse_transform(
    model.cluster_centers_
)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    s=300,
    marker='X',
    color='red'
)

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("K-Means Clustering")

plt.grid(True)
plt.show()