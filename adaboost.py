import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import*
np.random.seed(42)
size = 200
age = np.random.randint(18,60,size)
income = np.random.randint(20000,100000,size)
score = np.random.randint(1,100,size)
purchase = [
    1 if income[i] > 50000 and score[i] > 50 else 0 # can also do  and age[i] < 40
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
y=df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size =0.3, random_state=42
)

model = AdaBoostClassifier(n_estimators = 100,random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:", confusion_matrix(y_test, pred))
print("\nPrecision:", precision_score(y_test, pred))
print("\nRecall:", recall_score(y_test, pred))

test = pd.DataFrame({
    'Age': [42],
    'Income': [80000],
    'Score': [23]
})

result = model.predict(test)
print("Predicted Purchase:", "yes" if result[0] ==1 else 'no')