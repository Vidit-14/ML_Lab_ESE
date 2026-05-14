import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = 2 * np.random.rand(100,1)

y = 4 + 3 * X + np.random.rand(100,1)

df = pd.DataFrame({
    'X': X.flatten(),
    'y': y.flatten()
})

print("\nFirst 10 Rows of Dataset:\n")

print(df.head(10))

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("intercept", model.intercept_)
print("slope:", model.coef_)
print("RMSE:", rmse)

while True:
    try:
        user_input = float(input("Enter a value for X to predict y "))
        user_prediction = model.predict([[user_input]])   
        print(f"Predicted y for X = {user_input}: {user_prediction[0][0]}")
    except ValueError:
        print("Exiting...")
        break                                                                   