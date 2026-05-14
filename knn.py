import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import*
import matplotlib.pyplot as plt

np.random.seed(42)
size =100
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

X = df.drop('Purchase', axis=1)
y = df['Purchase']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_acc = 0
best_k = 1
accuracy_values = []

for k in range(1,21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracy_values.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_k = k
print("\nBest K:", best_k)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nAccuracy;", accuracy_score(y_test,pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nPrecision:", precision_score(y_test, pred))
print("\nRecall:", recall_score(y_test, pred))
print("\nF1 Score:", f1_score(y_test, pred))

# -------- PLOT --------

plt.figure(figsize=(8,5))

plt.plot(
    range(1, 21),
    accuracy_values,
    marker='o'
)

plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")

plt.grid(True)

plt.show()

test = [[30,70000,80]]
test = scaler.transform(test)
result = model.predict(test)
print("\npredicted Purchase:", "yes" if result[0]==1 else "no")