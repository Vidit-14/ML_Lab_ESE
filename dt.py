import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import*
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Age':['Young','Young','Middle','Old','Old','Old','Middle','Young','Young','Old'],
    'Income':['High','High','High','Medium','Low','Low','Low','Medium','Low','Medium'],
    'Student':['No','No','No','No','Yes','Yes','Yes','No','Yes','Yes'],
    'Credit':['Fair','Excellent','Fair','Fair','Fair','Excellent','Excellent','Fair','Fair','Fair'],
    'Buy':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes']
})

print(df)

enc = LabelEncoder()

for col in df.columns:
    df[col] = enc.fit_transform(df[col])

print(df)

X = df.drop("Buy", axis=1)
y = df["Buy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
# Metrics

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nPrecision:", precision_score(y_test, pred))
print("\nRecall:", recall_score(y_test, pred))
print("\nF1 Score:", f1_score(y_test, pred))

# -------- DECISION TREE PLOT --------

plt.figure(figsize=(10,6))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)

plt.title("Decision Tree Classifier")
plt.show()

# Test prediction
test = [[1, 1, 1, 0]]   # Young, Medium, Yes, Excellent

result = model.predict(test)

print("\nPredicted Output:",
      "Yes" if result[0] == 1 else "No")