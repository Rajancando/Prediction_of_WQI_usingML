
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('res_wqi.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

"""**splitting the dataset into training and test set**




"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

"""## Training the Decision Tree Regression model on the training dataset"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

print(X)

y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)

"""## Predicting a test result"""

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)