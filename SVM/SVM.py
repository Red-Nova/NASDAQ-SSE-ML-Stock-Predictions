from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('AAPL.csv')

# Changes The Date column as index columns
df.index = pd.to_datetime(df['Date'])

# Drop The original date column
df = df.drop(['Date'], axis='columns')

# Create predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]
X.head()

# Target variables: buy for +1, no position for 0
Y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# print(Y)

split_percentage = 0.8
split = int(split_percentage*len(df))

# Train data set
X_train = X[:split]
Y_train = Y[:split]

# Test data set
X_test = X[split:]
Y_test = Y[split:]

# Support vector classifier
cls = SVC().fit(X_train, Y_train)

# TODO Accuracy test
df['Predicted_Signal'] = cls.predict(X)

# Calculate daily returns
df['Return'] = df.Close.pct_change()

# Calculate strategy returns
df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)

# Calculate Cumulutive returns
df['Cum_Return'] = df['Return'].cumsum()

# Plot Strategy Cumulative returns
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()


plt.plot(df['Cum_Return'],color='red')
plt.plot(df['Cum_Strategy'],color='blue')