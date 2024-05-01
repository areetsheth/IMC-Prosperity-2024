import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('../data/prices_round_3_day_0.csv', delimiter=';')
# df_1 = pd.read_csv('../data/prices_round_3_day_1.csv', delimiter=';')
# df_2 = pd.read_csv('../data/prices_round_3_day_2.csv', delimiter=';')
# df = pd.concat([df, df_1, df_2], ignore_index=True)
#print(df)

def model_details(df, product_name):
    # Filter the DataFrame for the selected product
    df_product = df[df['product'] == product_name]
    
    prices = df_product["mid_price"].tolist()

    if product_name == "GIFT_BASKET":
        print(df_product["mid_price"].mean() ,df_product["mid_price"].std())
    # Using the last 4 points for training, if there are at least 4 points
    if len(df_product) >= 4:
        X = []
        Y = []
        # Populate the input features and target values using a sliding window approach
        window_size = 4
        for i in range(len(prices) - window_size):
            window = prices[i:i+window_size]
            target = prices[i+window_size]
            X.append(window)
            Y.append(target)
        model = LinearRegression()
        model.fit(X, Y)
        coefficients = model.coef_
        intercept = model.intercept_
        return {'coefficients': coefficients, 'intercept': intercept}
    else:
        return {'coefficients': None, 'intercept': None}  # Not enough data to train the model




chocolates = model_details(df, 'CHOCOLATE')
print("chocolate Coefficients (Weights):", chocolates['coefficients'])
print("chocolate Bias (Intercept):", chocolates['intercept'])
print()
roses = model_details(df, 'ROSES')
print("rose Coefficients (Weights):", roses['coefficients'])
print("rose Bias (Intercept):", roses['intercept'])
print()
straw = model_details(df, 'STRAWBERRIES')
print("strawberry Coefficients (Weights):", straw['coefficients'])
print("strawberry Bias (Intercept):", straw['intercept'])
print()
gift = model_details(df, 'GIFT_BASKET')
print("gift basket Coefficients (Weights):", gift['coefficients'])
print("gift basket Bias (Intercept):", gift['intercept'])

# Create a pivot table to get the mid_price for each product at each timestamp
pivot_table = df.pivot_table(index='timestamp', columns='product', values='mid_price')

# Calculate the new column as per the given formula
pivot_table['basket_comparison'] = (
    pivot_table['GIFT_BASKET']
    - 4 * pivot_table['CHOCOLATE']
    - 6 * pivot_table['STRAWBERRIES']
    - pivot_table['ROSES']
)

# Resetting index to bring the timestamp back as a column
# result = pivot_table.reset_index()[['timestamp', 'basket_comparison']]

# print(result["basket_comparison"].mean(), result["basket_comparison"].std())

# print(result)
# # Display the result
# print(result)
# plt.figure(figsize=(10, 6))
# plt.plot(result['timestamp'], result['basket_comparison'], label='Original Prices', color='blue')
# plt.xlabel('Timestamp')
# plt.ylabel('Price')
# plt.title('Basket Comparison Prices')
# plt.legend()
# plt.grid(True)
# plt.show()




