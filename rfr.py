
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *

np.random.seed(42)
size = 300
area = np.random.uniform(500,5000, size)
bedrooms = np.random.randint(1,5,size)
age = np.random.randint(1,30,size)

price =(area* 150 + bedrooms*10000 - age * 2000 + np.random.randint(-0.5, 5, size))


df = pd.DataFrame({
    'Area':area,
    'Bedrooms': bedrooms,
    'House_Age': age,
    'Price': price

})

print(df.head())
print(
    "Price Range:",
    df['Price'].min(),
    "to",
    df['Price'].max()
)
X = df.drop('Price', axis =1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.3, random_state =24
)   

model = RandomForestRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n Mean Absolute Error: ", mean_absolute_error(y_test, pred))
print("\n Mean Squared Error: ", mean_squared_error(y_test, pred))
print("\n R^2 Score: ", r2_score(y_test, pred))

# test = [[2500.6, 3, 11]]
test = pd.DataFrame({
    'Area': [2500.6],
    'Bedrooms': [3],
    'House_Age': [11]
})
result = model.predict(test)
print("\nPredicted House Price:",
      round(result[0], 2))