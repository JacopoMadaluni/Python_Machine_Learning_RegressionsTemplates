# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 01:30:48 2018

@author: Jaco
"""

# Decision Tree Regression
# Disclaimer: this regression model is using a two dimension dataset.
# An optimal use case for this template is a at least three dimension datasets. 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1 : -1].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



# Fitting Decision Tree Regression  to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor( splitter = "best", random_state = 42)
regressor.fit(X,y)



# Predict a new result 
y_pred = regressor.predict(6.5)


# Visualising the Decision Tree Regression prediction (higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Decision Tree Prediction")
plt.show()
