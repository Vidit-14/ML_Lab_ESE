import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

np.random.seed(42)

size = 200

# Generate synthetic data
age = np.random.randint(18, 60, size)
income = np.random.randint(20000, 100000, size)
score = np.random.randint(1, 100, size)

# Binary classification target
purchase = [
    1 if income[i] > 50000 and score[i] > 50 else 0
    for i in range(size)
]

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Score': score,
    'Purchase': purchase
})

print(df.head())

# Features and Target
X = df.drop('Purchase', axis=1)
y = df['Purchase']

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

# Logistic Regression Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
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

# Probability Predictions
prob = model.predict_proba(X_test)

print("\nFirst 5 Probability Predictions:\n")
print(prob[:5])

# -------- VISUALIZATION --------

plt.figure(figsize=(8,6))

plt.scatter(
    X_test[:,1],   # Income
    X_test[:,2],   # Score
    c=pred
)

plt.xlabel("Scaled Income")
plt.ylabel("Scaled Score")
plt.title("Logistic Regression Classification")

plt.grid(True)

plt.show()

# -------------------------------

# Test Prediction
test = [[30, 70000, 80]]

test = scaler.transform(test)

result = model.predict(test)

probability = model.predict_proba(test)

print("\nPredicted Purchase:",
      "Yes" if result[0] == 1 else "No")

print("\nPrediction Probability:")
print(probability)