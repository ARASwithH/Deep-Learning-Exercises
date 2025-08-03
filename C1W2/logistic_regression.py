import numpy as np
import matplotlib.pyplot as plt
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




# Forward and Backward propagation

'''
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
'''
def propagation(w,b,X,Y):
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    m = X.shape[1]
    cost = np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m
    dw = np.dot(X, (A-Y).T) / m
    db = np.sum(A-Y) / m

    return cost, dw, db


'''
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
'''
def optimazer(w, b, X, Y, num_iterations, learning_rate):

    costs = []
    
    for i in range(num_iterations):

        cost, dw, db = propagation(w, b, X, Y)

        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)

        # to visualize cost changes
        if i % 100 == 0 :
            costs.append(cost)

    return w, b, cost, costs


'''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
'''
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A>=0.5)*1.0




# Merge all functions into a model
'''
Builds the logistic regression model by calling the function you've implemented previously

Arguments:
X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
print_cost -- Set to true to print the cost every 100 iterations

Returns:
d -- dictionary containing information about the model.
'''

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):

    w, b = params_initializer(X_train.shape[0])
    w, b, cost, costs = optimazer(w, b, X_train, Y_train, num_iterations, learning_rate)
    
    Y_train_prediction = predict(w, b, X_train)
    Y_test_prediction = predict(w, b, X_test)

    return {
        'predicted_train_Y' : Y_train_prediction,
        'predicted_test_Y' : Y_test_prediction,
        'best_w' : w,
        'best_b' : b,
        'final_cost' : cost,
        'costs' : costs
    }

    

info = model(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, 4000, 0.005)

# Model Validation

print("train accuracy: {} %".format(100 - np.mean(np.abs(info['predicted_train_Y'] - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(info['predicted_test_Y'] - test_set_y)) * 100))

plt.plot(info["costs"])
plt.ylabel("cost")
plt.xlabel("iterations (x100)")
plt.title("Learning rate = 0.05")
plt.show()








