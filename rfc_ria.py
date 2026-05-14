# Simple Random Forest Classification Example

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

# Create synthetic dataset
np.random.seed(42)

size = 200

age = np.random.randint(18, 60, size)
income = np.random.randint(20000, 100000, size)
score = np.random.randint(1, 100, size)

purchase = [
    1 if income[i] > 50000 and score[i] > 50 else 0
    for i in range(size)
]

df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Score': score,
    'Purchase': purchase
})

print(df.head())

# Features and target
X = df.drop('Purchase', axis=1)
y = df['Purchase']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Metrics
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nPrecision:", precision_score(y_test, pred))
print("\nRecall:", recall_score(y_test, pred))
print("\nF1 Score:", f1_score(y_test, pred))

# Test prediction
test = [[30, 75000, 85]]

result = model.predict(test)

print("\nPredicted Purchase:",
      "Yes" if result[0] == 1 else "No")