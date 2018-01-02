import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('1-brain_body.txt') # fwf: fixed width file
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# train model on data
body_reg =linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
#%matplotlib inline
#plt.scatter(x_values, y_values)
plt.scatter(dataframe['Brain'], dataframe['Body'])
plt.plot(dataframe['Brain'], body_reg.predict(x_values))
plt.show()