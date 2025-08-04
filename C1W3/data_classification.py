import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.datasets
from sklearn.metrics import classification_report
import sklearn.linear_model
from planar_utils import (
    plot_decision_boundary,
    sigmoid,
    load_planar_dataset,
    load_extra_datasets,
)

np.random.seed(1)  # set a seed so that the results are consistent




# Dataset

X, Y = load_planar_dataset()

'''
**Exercise**: How many training examples do you have? In addition, what is the `shape` of the variables `X` and `Y`? 
**Hint**: How do you get the shape of a numpy array? [(help)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html)
'''
X_shape = X.shape
Y_shape = Y.shape
m = X_shape[1]




# Simple Logistic Regression
'''
Before building a full neural network, lets first see how logistic regression performs on this problem.
You can use sklearn's built-in functions to do that.
Run the code below to train a logistic regression classifier on the dataset.
'''
lr = sklearn.linear_model.LogisticRegressionCV()
X_train, X_test, y_train, y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
lr.fit(X_train, y_train.ravel())
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))





