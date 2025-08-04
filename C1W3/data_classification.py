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





