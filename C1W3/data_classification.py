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




# Neural Network model

# Defining the neural network structure
'''
**Exercise**: Define three variables:
    - n_x: the size of the input layer
    - n_h: the size of the hidden layer (set this to 4) 
    - n_y: the size of the output layer

**Hint**: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.
'''
def define_layer_size(X, Y):
    return X.shape[0] , 4, Y.shape[0]


# Initialize the model's parameters
'''
**Exercise**: Implement the function `initialize_parameters()`.

**Instructions**:
- Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
- You will initialize the weights matrices with random values. 
    - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
- You will initialize the bias vectors as zeros. 
    - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.
'''
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(n_h, n_y)
    w2 = np.random.randn(n_h, n_y) * 0.01
    b2 = np.zeros(n_y, n_y)
    
    return w1, b1, w2, b2









