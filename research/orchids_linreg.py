import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('../data/prices_round_2_day_0.csv', delimiter=';')

prices = df["ORCHIDS"].tolist()
# Initialize lists to store input features (X) and target values (Y)
X = []
Y = []

# Populate the input features and target values using a sliding window approach
window_size = 4
for i in range(len(prices) - window_size):
    window = prices[i:i+window_size]
    target = prices[i+window_size]
    X.append(window)
    Y.append(target)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, Y)

# Extract the coefficients (weights) and bias (intercept) from the model
coefficients = model.coef_
bias = model.intercept_

print("Coefficients (Weights):", coefficients)
print("Bias (Intercept):", bias)
