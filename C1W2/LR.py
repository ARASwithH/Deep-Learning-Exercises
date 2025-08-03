import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()



# Overview of the Problem set

'''
**Exercise:** Find the values for:
    - m_train (number of training examples)
    - m_test (number of test examples)
    - num_px (= height = width of a training image)
Remember that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3).
For instance, you can access `m_train` by writing `train_set_x_orig.shape[0]`.
'''
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]



'''
**Exercise:** Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num\_px $*$ num\_px $*$ 3, 1).
A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use: 

X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
'''
train_set_x_flatten = train_set_x_orig.reshape(num_px*num_px*3, m_train) / 255
test_set_x_flatten = test_set_x_orig.reshape(num_px*num_px*3, m_test) / 255




# Building the parts of our algorithm

'''
**Exercise**:implement `sigmoid()`.
As you've seen in the figure above, you need to compute $sigmoid( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$ to make predictions.
Use np.exp().
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



'''
**Exercise:** Implement parameter initialization in the cell below. You have to initialize w as a vector of zeros.
If you don't know what numpy function to use, look up np.zeros() in the Numpy library's documentation.
'''
def params_initializer(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b





