import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


dataframe = pd.read_fwf('brain_body.txt')
x_axis = dataframe[['Brain']]
y_axis = dataframe[['Body']]

body_model = linear_model.LinearRegression()
body_model.fit(x_axis,y_axis)


plt.scatter(x_axis,y_axis)
plt.plot(x_axis,body_model.predict(x_axis))

plt.show()