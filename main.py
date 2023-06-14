import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


df_milk = pd.read_csv('Milk production.csv', sep = ",", index_col='Month',parse_dates=True)
df_milk
print(df_milk.describe())
print(df_milk.head())

# graph serie temporelle axe 1
fig, ax1 = plt.subplots()
df_milk.plot(ax=ax1, figsize=(12,10))
plt.title('production of milk')
plt.xlabel('Date')
plt.ylabel('Production')
plt.show()


# Split the data into a train and test set
t_train = df_milk.loc[:'1970']
t_test = df_milk.loc['1971':]

# Create an axis
fig, ax = plt.subplots()

# Plot the train and test sets on the axis ax
t_train.plot(ax=ax, figsize=(12,10))
t_test.plot(ax=ax)
plt.title('train - test split of the production of milk')
plt.xlabel('Date')
plt.ylabel('fabrication')
plt.show()

# # Moyenne Mobile
# df_milk_mean = milk.rolling(window=10).mean()
# fig, ax2 = plt.subplots()
# df_milk_mean.plot(ax=ax2, figsize=(12,10))
# plt.title('Milk production')
# plt.xlabel('Date')
# plt.ylabel('Production')
# plt.show()
 

# # les valeurs reelles et forecast
# milk_base = pd.concat([milk, milk.shift(1)],axis=1)
# print(milk_base)
# milk_base.columns = ['actual_sales', 'forecast_sales']
# print(milk_base)
# print(milk_base.dropna())


# # MSE
# milk_error = mean_squared_error(milk_base.actual_sales, milk_base.forecast_sales)
# print(milk_error)













