"""##importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('res_wqi.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

print(X)

[[6.83e+00 3.11e+01 8.00e+01 5.00e+00 6.00e+01 6.80e-02 1.36e+02 6.01e+00]
 [6.52e+00 3.11e+01 3.00e+01 2.50e+00 5.00e+01 4.30e-02 8.60e+01 6.25e+00]
 [6.36e+00 3.12e+01 4.00e+01 2.00e+00 5.40e+01 8.00e-03 1.70e+01 7.01e+00]
 [6.18e+00 3.04e+01 4.00e+01 4.50e+00 8.00e+01 3.40e-02 6.80e+01 4.33e+00]
 [6.05e+00 3.05e+01 6.40e+01 5.50e+00 6.00e+01 3.10e-02 6.30e+01 5.12e+00]
 [6.16e+00 3.05e+01 5.60e+01 3.00e+00 6.00e+01 3.00e-02 6.10e+01 7.00e+00]
 [6.18e+00 3.12e+01 6.00e+01 7.50e+00 7.60e+01 7.10e-02 1.41e+02 7.57e+00]
 [6.38e+00 3.10e+01 2.00e+02 2.50e+00 1.24e+02 4.70e-02 9.40e+01 6.56e+00]
 [6.90e+00 3.21e+01 3.20e+01 3.00e+00 7.20e+01 3.70e-02 7.50e+01 8.55e+00]
 [6.20e+00 3.11e+01 4.40e+01 2.00e+00 4.00e+01 3.80e-02 7.50e+01 5.16e+00]
 [6.06e+00 3.13e+01 1.60e+01 2.00e+00 1.00e+02 3.30e-02 6.70e+01 5.89e+00]
 [6.16e+00 3.14e+01 4.40e+01 1.00e+00 4.00e+01 2.90e-02 5.80e+01 7.78e+00]
 [6.39e+00 3.15e+01 3.60e+01 2.00e+00 4.80e+01 5.90e-02 1.18e+02 6.04e+00]
 [6.75e+00 3.19e+01 3.20e+01 3.00e+00 3.60e+01 4.30e-02 8.50e+01 6.61e+00]
 [6.20e+00 3.13e+01 2.80e+01 1.00e+00 4.00e+01 3.30e-02 6.60e+01 5.69e+00]
 [6.09e+00 3.17e+01 2.40e+01 3.00e+00 4.80e+01 2.90e-02 5.90e+01 6.14e+00]
 [6.06e+00 3.14e+01 4.00e+01 2.00e+00 4.40e+01 2.80e-02 5.70e+01 7.38e+00]
 [6.87e+00 3.17e+01 2.40e+01 3.00e+00 4.40e+01 4.10e-02 8.30e+01 6.76e+00]
 [6.07e+00 3.14e+01 5.40e+01 2.50e+00 4.60e+01 6.30e-02 1.22e+02 6.09e+00]
 [6.05e+00 3.11e+01 4.40e+01 3.00e+00 4.00e+01 5.40e-02 8.30e+01 6.26e+00]
 [6.92e+00 3.19e+01 4.00e+01 3.00e+00 4.40e+01 4.70e-02 7.20e+01 7.04e+00]
 [6.22e+00 3.12e+01 3.20e+01 2.00e+00 3.80e+01 3.50e-02 7.30e+01 6.16e+00]
 [6.08e+00 3.14e+01 4.00e+01 3.00e+00 4.80e+01 3.20e-02 6.50e+01 7.24e+00]
 [6.15e+00 3.16e+01 5.60e+01 2.50e+00 8.00e+01 2.80e-02 5.40e+01 6.71e+00]
 [5.94e+00 2.74e+01 7.60e+01 5.00e+00 6.00e+01 5.60e-02 1.12e+02 4.74e+00]
 [6.28e+00 2.77e+01 7.60e+01 4.00e+00 4.00e+01 6.00e-02 1.20e+02 4.91e+00]
 [5.55e+00 2.75e+01 2.80e+01 5.00e+00 5.00e+01 5.80e-02 1.17e+02 4.49e+00]
 [5.94e+00 2.89e+01 2.80e+01 5.00e+00 3.80e+01 4.40e-02 8.80e+01 4.05e+00]
 [5.96e+00 2.92e+01 5.20e+01 3.00e+00 4.00e+01 3.80e-02 7.70e+01 4.14e+00]
 [5.34e+00 2.83e+01 2.40e+01 7.00e+00 8.00e+01 3.20e-02 6.40e+01 4.09e+00]
 [4.95e+00 2.75e+01 3.60e+01 9.00e+00 6.00e+01 8.00e-02 1.60e+02 4.53e+00]
 [5.08e+00 2.76e+01 2.80e+01 2.00e+00 1.10e+02 7.10e-02 1.42e+02 4.66e+00]
 [5.70e+00 2.87e+01 3.60e+01 8.00e+00 2.40e+02 5.80e-02 1.16e+02 4.59e+00]
 [5.19e+00 2.83e+01 2.80e+01 6.00e+00 5.40e+01 2.90e-02 5.90e+01 4.47e+00]
 [5.12e+00 2.82e+01 2.80e+01 7.00e+00 1.20e+02 2.90e-02 5.90e+01 4.52e+00]
 [6.04e+00 2.76e+01 3.20e+01 7.00e+00 1.24e+02 6.70e-02 1.35e+02 5.87e+00]]

print(y)

[10.22  8.53  8.4   6.45  7.06  8.79 10.82 11.86 11.98  6.97  7.82  9.28
  8.68  9.23  7.2   7.56  8.75  9.47  8.52  7.94  9.96  7.78  8.8   8.88
  6.86  7.28  5.65  5.36  5.65  4.96  5.71  6.14  8.31  4.8   5.46  8.49]

"""**splitting the dataset into training and test set**"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


print(X_train)

[[5.34e+00 2.83e+01 2.40e+01 7.00e+00 8.00e+01 3.20e-02 6.40e+01 4.09e+00]
 [5.55e+00 2.75e+01 2.80e+01 5.00e+00 5.00e+01 5.80e-02 1.17e+02 4.49e+00]
 [5.94e+00 2.89e+01 2.80e+01 5.00e+00 3.80e+01 4.40e-02 8.80e+01 4.05e+00]
 [5.19e+00 2.83e+01 2.80e+01 6.00e+00 5.40e+01 2.90e-02 5.90e+01 4.47e+00]
 [5.94e+00 2.74e+01 7.60e+01 5.00e+00 6.00e+01 5.60e-02 1.12e+02 4.74e+00]
 [6.28e+00 2.77e+01 7.60e+01 4.00e+00 4.00e+01 6.00e-02 1.20e+02 4.91e+00]
 [6.05e+00 3.05e+01 6.40e+01 5.50e+00 6.00e+01 3.10e-02 6.30e+01 5.12e+00]
 [6.36e+00 3.12e+01 4.00e+01 2.00e+00 5.40e+01 8.00e-03 1.70e+01 7.01e+00]
 [6.20e+00 3.13e+01 2.80e+01 1.00e+00 4.00e+01 3.30e-02 6.60e+01 5.69e+00]
 [6.06e+00 3.13e+01 1.60e+01 2.00e+00 1.00e+02 3.30e-02 6.70e+01 5.89e+00]
 [5.70e+00 2.87e+01 3.60e+01 8.00e+00 2.40e+02 5.80e-02 1.16e+02 4.59e+00]
 [6.08e+00 3.14e+01 4.00e+01 3.00e+00 4.80e+01 3.20e-02 6.50e+01 7.24e+00]
 [5.08e+00 2.76e+01 2.80e+01 2.00e+00 1.10e+02 7.10e-02 1.42e+02 4.66e+00]
 [6.92e+00 3.19e+01 4.00e+01 3.00e+00 4.40e+01 4.70e-02 7.20e+01 7.04e+00]
 [6.07e+00 3.14e+01 5.40e+01 2.50e+00 4.60e+01 6.30e-02 1.22e+02 6.09e+00]
 [6.18e+00 3.12e+01 6.00e+01 7.50e+00 7.60e+01 7.10e-02 1.41e+02 7.57e+00]
 [6.75e+00 3.19e+01 3.20e+01 3.00e+00 3.60e+01 4.30e-02 8.50e+01 6.61e+00]
 [6.38e+00 3.10e+01 2.00e+02 2.50e+00 1.24e+02 4.70e-02 9.40e+01 6.56e+00]
 [6.04e+00 2.76e+01 3.20e+01 7.00e+00 1.24e+02 6.70e-02 1.35e+02 5.87e+00]
 [6.52e+00 3.11e+01 3.00e+01 2.50e+00 5.00e+01 4.30e-02 8.60e+01 6.25e+00]
 [6.06e+00 3.14e+01 4.00e+01 2.00e+00 4.40e+01 2.80e-02 5.70e+01 7.38e+00]
 [6.83e+00 3.11e+01 8.00e+01 5.00e+00 6.00e+01 6.80e-02 1.36e+02 6.01e+00]
 [6.09e+00 3.17e+01 2.40e+01 3.00e+00 4.80e+01 2.90e-02 5.90e+01 6.14e+00]
 [6.16e+00 3.05e+01 5.60e+01 3.00e+00 6.00e+01 3.00e-02 6.10e+01 7.00e+00]
 [6.16e+00 3.14e+01 4.40e+01 1.00e+00 4.00e+01 2.90e-02 5.80e+01 7.78e+00]
 [6.20e+00 3.11e+01 4.40e+01 2.00e+00 4.00e+01 3.80e-02 7.50e+01 5.16e+00]
 [6.90e+00 3.21e+01 3.20e+01 3.00e+00 7.20e+01 3.70e-02 7.50e+01 8.55e+00]
 [6.39e+00 3.15e+01 3.60e+01 2.00e+00 4.80e+01 5.90e-02 1.18e+02 6.04e+00]]


print(y_train)

[ 4.96  5.65  5.36  4.8   6.86  7.28  7.06  8.4   7.2   7.82  8.31  8.8
  6.14  9.96  8.52 10.82  9.23 11.86  8.49  8.53  8.75 10.22  7.56  8.79
  9.28  6.97 11.98  8.68]


print(X_test)

[[4.95e+00 2.75e+01 3.60e+01 9.00e+00 6.00e+01 8.00e-02 1.60e+02 4.53e+00]
 [5.12e+00 2.82e+01 2.80e+01 7.00e+00 1.20e+02 2.90e-02 5.90e+01 4.52e+00]
 [5.96e+00 2.92e+01 5.20e+01 3.00e+00 4.00e+01 3.80e-02 7.70e+01 4.14e+00]
 [6.18e+00 3.04e+01 4.00e+01 4.50e+00 8.00e+01 3.40e-02 6.80e+01 4.33e+00]
 [6.05e+00 3.11e+01 4.40e+01 3.00e+00 4.00e+01 5.40e-02 8.30e+01 6.26e+00]
 [6.87e+00 3.17e+01 2.40e+01 3.00e+00 4.40e+01 4.10e-02 8.30e+01 6.76e+00]
 [6.22e+00 3.12e+01 3.20e+01 2.00e+00 3.80e+01 3.50e-02 7.30e+01 6.16e+00]
 [6.15e+00 3.16e+01 5.60e+01 2.50e+00 8.00e+01 2.80e-02 5.40e+01 6.71e+00]]


print(y_test)

[5.71 5.46 5.65 6.45 7.94 9.47 7.78 8.88]

"""## Training the Decision Tree Regression model on the training dataset"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

"""##predicting the y_pred data with test data """

y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
[[5.65 5.71]
 [4.8  5.46]
 [5.36 5.65]
 [4.96 6.45]
 [9.23 7.94]
 [7.56 9.47]
 [9.28 7.78]
 [8.79 8.88]]

""" ##calculate rmse value"""

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
1.1339804369976816

"""## Predicting a test result"""

from sklearn.metrics import r2_score
res=r2_score(y_test,y_pred)
print(res)
0.39764487871247356
