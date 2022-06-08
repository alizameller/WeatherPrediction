import pandas as pd
from statsmodels.tsa.ar_model import AutoReg as AR
import matplotlib.pyplot as plt
import numpy as np


weather = pd.read_csv("Weather_Dataset.csv", parse_dates=["Date"], index_col="Date")
w = weather.values
train = w[0:5845]  # data for training
test = w[5845:7671]  # data for testing
predictions = [] 

model_ar = AR(train, lags=1800)
model_ar_fit = model_ar.fit()
predictions = model_ar_fit.predict(start=5845, end=7671)

dates = pd.read_csv("Weather_Dataset.csv", usecols=["Date"]).values
plt.plot(test)
plt.xticks(np.arange(1826), dates[5845:7671])  # Set text labels.
plt.plot(predictions, color='red')  # predicted data
plt.show()
