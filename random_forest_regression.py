import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('res_wqi.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

"""### **splitting the datset into training set and test set**"""

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

"""## Training the Random Forest Regression model on the training dataset"""

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, Y_train)

"""# **predicting the test result**"""

Y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)

"""## Visualising the Random Forest Regression results (higher resolution)"""

from sklearn.metrics import r2_score
r2_score(y_test, Y_pred)