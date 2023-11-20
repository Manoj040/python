import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset (replace this with your own dataset)
# Assuming the dataset has columns: 'Month', 'Price', 'Temperature', 'Rainfall'
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Price': [10, 7, 19, 26, 15, 30, 16, 39, 19, 32],
    'Temperature': [25, 48, 28, 28, 32, 15, 9, 50, 10, 2],
    'Rainfall': [38, 45, 49, 27, 19, 25, 20, 22, 28, 30]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Separating features (X) and target variable (y)
X = df[['Month', 'Temperature', 'Rainfall']]
y = df['Price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions for the next 3 months
future_months = np.array([[11, 9, 28], [10, 26, 3], [13, 25, 35]])  # Replace with your own future data
future_prices = model.predict(future_months)

# Displaying the predictions
for month, price in zip(future_months[:, 0], future_prices):
    print(f"Predicted price for Month {month}: ${price:.2f}")

# Visualizing the results
plt.scatter(X['Month'], y, color='blue', label='Actual Prices')
plt.plot(X['Month'], model.predict(X), color='red', label='Linear Regression')
plt.scatter(future_months[:, 0], future_prices, color='green', label='Predicted Prices')
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Vegetable Price Prediction')
plt.legend()
plt.show()
