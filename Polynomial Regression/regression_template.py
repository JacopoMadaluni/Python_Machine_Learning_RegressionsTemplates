# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 02:38:32 2018

@author: Jaco
"""

# Regression Template

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 01:39:56 2018

@author: Jaco
"""

# Polynomial Regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('<file>.csv')
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



# Fitting Regression Model to the dataset
# Create the regressor here 
 



# Predict a new result 
y_pred = regressor.predict(6.5)


# Visualising the Regression prediction
plt.scatter(X, y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Polynomial Prediction")
plt.show()


# Visualising the Regression prediction (higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Polynomial Prediction")
plt.show()



