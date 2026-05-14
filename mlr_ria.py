import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

size = 500

med_inc = np.random.uniform(1, 15, size)
house_age = np.random.randint(1,50, size)
ave_rooms = np.random.uniform(2, 10, size)
ave_bedrms = np.random.uniform(1, 5, size)
population = np.random.randint(100, 5000, size)
ave_occup = np.random.uniform(1, 6, size)
latitude = np.random.uniform(32, 42, size)
longitude = np.random.uniform(-124, -114, size)

price = (
    0.5 * med_inc +
    0.01 * house_age +
    0.2 * ave_rooms -
    0.1 * ave_bedrms -
    0.0001 * population -
    0.05 * ave_occup -
    0.02 * latitude -
    0.03 * longitude +
    np.random.normal(0, 0.5, size)
)

df = pd.DataFrame({
    'MedInc': med_inc,
    'HouseAge': house_age,
    'AveRooms': ave_rooms,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': ave_occup,
    'Latitude': latitude,
    'Longitude': longitude,
    'Price': price
})

print("First 5 Rows of Synthetic Dataset:\n")

print(df.head())

X = df.drop('Price', axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("RMSE:", rmse)

test_data = pd.DataFrame([{
    'MedInc': 5.0,
    'HouseAge': 20.0,
    'AveRooms': 5.5,
    'AveBedrms': 1.1,
    'Population': 1500.0,
    'AveOccup': 3.0,
    'Latitude': 34.05,
    'Longitude': -118.25
}])

prediction = model.predict(test_data)
# Convert to dollars
predicted_price = prediction[0] * 100000
# Display Results
print("\nTest Data:\n")
print(test_data)
print(f"\nPredicted Value (Dataset Units): {prediction[0]:.4f}")
print(f"Approx House Price: ${predicted_price:,.2f}")