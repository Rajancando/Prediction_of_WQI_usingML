import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

"""## Importing the dataset"""

dataset=pd.read_csv('res_wqi.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

print(x)
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
  6.86  7.28  5.65  5.36  5.65  4.96 ]
"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print (x_train)

[[6.16e+00 3.14e+01 4.40e+01 1.00e+00 4.00e+01 2.90e-02 5.80e+01 7.78e+00]
 [5.34e+00 2.83e+01 2.40e+01 7.00e+00 8.00e+01 3.20e-02 6.40e+01 4.09e+00]
 [5.94e+00 2.89e+01 2.80e+01 5.00e+00 3.80e+01 4.40e-02 8.80e+01 4.05e+00]
 [6.04e+00 2.76e+01 3.20e+01 7.00e+00 1.24e+02 6.70e-02 1.35e+02 5.87e+00]
 [5.19e+00 2.83e+01 2.80e+01 6.00e+00 5.40e+01 2.90e-02 5.90e+01 4.47e+00]
 [5.96e+00 2.92e+01 5.20e+01 3.00e+00 4.00e+01 3.80e-02 7.70e+01 4.14e+00]
 [5.70e+00 2.87e+01 3.60e+01 8.00e+00 2.40e+02 5.80e-02 1.16e+02 4.59e+00]
 [6.90e+00 3.21e+01 3.20e+01 3.00e+00 7.20e+01 3.70e-02 7.50e+01 8.55e+00]
 [6.75e+00 3.19e+01 3.20e+01 3.00e+00 3.60e+01 4.30e-02 8.50e+01 6.61e+00]
 [6.16e+00 3.05e+01 5.60e+01 3.00e+00 6.00e+01 3.00e-02 6.10e+01 7.00e+00]
 [6.87e+00 3.17e+01 2.40e+01 3.00e+00 4.40e+01 4.10e-02 8.30e+01 6.76e+00]
 [6.20e+00 3.13e+01 2.80e+01 1.00e+00 4.00e+01 3.30e-02 6.60e+01 5.69e+00]
 [6.38e+00 3.10e+01 2.00e+02 2.50e+00 1.24e+02 4.70e-02 9.40e+01 6.56e+00]
 [5.55e+00 2.75e+01 2.80e+01 5.00e+00 5.00e+01 5.80e-02 1.17e+02 4.49e+00]
 [6.52e+00 3.11e+01 3.00e+01 2.50e+00 5.00e+01 4.30e-02 8.60e+01 6.25e+00]
 [6.39e+00 3.15e+01 3.60e+01 2.00e+00 4.80e+01 5.90e-02 1.18e+02 6.04e+00]
 [6.28e+00 2.77e+01 7.60e+01 4.00e+00 4.00e+01 6.00e-02 1.20e+02 4.91e+00]
 [5.94e+00 2.74e+01 7.60e+01 5.00e+00 6.00e+01 5.60e-02 1.12e+02 4.74e+00]
 [6.18e+00 3.12e+01 6.00e+01 7.50e+00 7.60e+01 7.10e-02 1.41e+02 7.57e+00]
 [6.15e+00 3.16e+01 5.60e+01 2.50e+00 8.00e+01 2.80e-02 5.40e+01 6.71e+00]
 [6.05e+00 3.05e+01 6.40e+01 5.50e+00 6.00e+01 3.10e-02 6.30e+01 5.12e+00]
 [6.07e+00 3.14e+01 5.40e+01 2.50e+00 4.60e+01 6.30e-02 1.22e+02 6.09e+00]
 [6.22e+00 3.12e+01 3.20e+01 2.00e+00 3.80e+01 3.50e-02 7.30e+01 6.16e+00]
 [6.05e+00 3.11e+01 4.40e+01 3.00e+00 4.00e+01 5.40e-02 8.30e+01 6.26e+00]
 [6.20e+00 3.11e+01 4.40e+01 2.00e+00 4.00e+01 3.80e-02 7.50e+01 5.16e+00]
 [5.12e+00 2.82e+01 2.80e+01 7.00e+00 1.20e+02 2.90e-02 5.90e+01 4.52e+00]
 [6.18e+00 3.04e+01 4.00e+01 4.50e+00 8.00e+01 3.40e-02 6.80e+01 4.33e+00]
 [6.83e+00 3.11e+01 8.00e+01 5.00e+00 6.00e+01 6.80e-02 1.36e+02 6.01e+00]]


print(y_train)

[ 9.28  4.96  5.36  8.49  4.8   5.65  8.31 11.98  9.23  8.79  9.47  7.2
 11.86  5.65  8.53  8.68  7.28  6.86 10.82  8.88  7.06  8.52  7.78  7.94
  6.97  5.46  6.45 10.22]


print(x_test)

[[5.08e+00 2.76e+01 2.80e+01 2.00e+00 1.10e+02 7.10e-02 1.42e+02 4.66e+00]
 [6.92e+00 3.19e+01 4.00e+01 3.00e+00 4.40e+01 4.70e-02 7.20e+01 7.04e+00]
 [6.06e+00 3.14e+01 4.00e+01 2.00e+00 4.40e+01 2.80e-02 5.70e+01 7.38e+00]
 [4.95e+00 2.75e+01 3.60e+01 9.00e+00 6.00e+01 8.00e-02 1.60e+02 4.53e+00]
 [6.08e+00 3.14e+01 4.00e+01 3.00e+00 4.80e+01 3.20e-02 6.50e+01 7.24e+00]
 [6.09e+00 3.17e+01 2.40e+01 3.00e+00 4.80e+01 2.90e-02 5.90e+01 6.14e+00]
 [6.06e+00 3.13e+01 1.60e+01 2.00e+00 1.00e+02 3.30e-02 6.70e+01 5.89e+00]
 [6.36e+00 3.12e+01 4.00e+01 2.00e+00 5.40e+01 8.00e-03 1.70e+01 7.01e+00]]

print(y_test)
[6.14 9.96 8.75 5.71 8.8  7.56 7.82 8.4 ]


"""## Training the Multiple Linear Regression model on the Training set"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


"""## Predicting the Test set results"""

y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

[[5.87 6.14]
 [9.8  9.96]
 [8.89 8.75]
 [5.65 5.71]
 [8.95 8.8 ]
 [7.67 7.56]
 [7.82 7.82]
 [8.58 8.4 ]]

**predict the result using rmse**

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
0.14901896462463984

"""## to calculate r2 score set"""

from sklearn.metrics import r2_score
result=r2_score(y_test, y_pred)
print (result)
0.9873724713212202
