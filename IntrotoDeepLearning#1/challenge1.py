import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np


dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[[0]]


y_values = dataframe[[1]]


body_model = linear_model.LinearRegression()
body_model.fit(x_values,y_values)

print(body_model.coef_)

print(np.mean((body_model.predict(x_values)-y_values) **2))

print(body_model.score(x_values,y_values))


plt.scatter(x_values,y_values)

plt.plot(x_values,body_model.predict(x_values))

plt.show()