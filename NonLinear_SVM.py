import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

np.random.seed(42)

size = 300

# Generate random data
x1 = np.random.randn(size)
x2 = np.random.randn(size)

# Create NON-LINEAR circular classes
y = []

for i in range(size):

    distance = x1[i]**2 + x2[i]**2

    if distance > 1.5:
        y.append(1)
    else:
        y.append(0)

df = pd.DataFrame({
    'Feature1': x1,
    'Feature2': x2,
    'Class': y
})

print(df.head())

X = df[['Feature1', 'Feature2']]
y = df['Class']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# NON-LINEAR SVM using RBF Kernel
model = SVC(kernel='rbf')

model.fit(X_train, y_train)

pred = model.predict(X_test)

# Metrics
print("\nAccuracy:",
      accuracy_score(y_test, pred))

print("\nConfusion Matrix:\n",
      confusion_matrix(y_test, pred))

print("\nPrecision:",
      precision_score(y_test, pred))

print("\nRecall:",
      recall_score(y_test, pred))

print("\nF1 Score:",
      f1_score(y_test, pred))

# -------- VISUALIZATION --------

plt.figure(figsize=(8,6))

plt.scatter(
    X_test[:,0],
    X_test[:,1],
    c=pred
)

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Non-Linear SVM using RBF Kernel")

plt.grid(True)

plt.show()

# -------------------------------

# Test Prediction
test = [[1.5, 1.5]]

test = scaler.transform(test)

result = model.predict(test)

print("\nPredicted Class:",
      "Class 1" if result[0] == 1 else "Class 0")