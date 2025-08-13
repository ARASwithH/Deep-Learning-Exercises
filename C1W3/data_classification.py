import numpy as np
import matplotlib.pyplot as plt
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
print('Logistic Regression report:')
print(classification_report(y_test, y_pred))




# Neural Network model functions

# Defining the neural network structure
'''
**Exercise**: Define three variables:
    - n_x: the size of the input layer
    - n_h: the size of the hidden layer (set this to 4) 
    - n_y: the size of the output layer

**Hint**: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.
'''
def define_layer_size(X, Y, n_h):
    return X.shape[0] , n_h, Y.shape[0]


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
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, n_y))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, n_y))

    parameters = {"W1": w1,
                  "b1": b1,
                  "W2": w2,
                  "b2": b2}
    
    return parameters


# The Loop
'''
**Question**: Implement `forward_propagation()`.

**Instructions**:
- Look above at the mathematical representation of your classifier.
- You can use the function `sigmoid()`. It is built-in (imported) in the notebook.
- You can use the function `np.tanh()`. It is part of the numpy library.
- The steps you have to implement are:
    1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
    2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
- Values needed in the backpropagation are stored in "`cache`". The `cache` will be given as an input to the backpropagation function.
'''
def forward_propagation(X, parameters):
    w1 = parameters["W1"]
    b1 = parameters["b1"]
    w2 = parameters["W2"]
    b2 = parameters["b2"]
    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    cache = {"Z1": z1,
             "A1": a1,
             "Z2": z2,
             "A2": a2}
    
    return cache


'''
**Exercise**: Implement `compute_cost()` to compute the value of the cost $J$.

**Instructions**:
- There are many ways to implement the cross-entropy loss. To help you, we give you how we would have implemented
$- \sum\limits_{i=0}^{m}  y^{(i)}\log(a^{[2](i)})$:
```python
logprobs = np.multiply(np.log(A2),Y)
cost = - np.sum(logprobs)                # no need to use a for loop!
```

(you can use either `np.multiply()` and then `np.sum()` or directly `np.dot()`).  
Note that if you use `np.multiply` followed by `np.sum` the end result will be a type `float`, whereas if you use `np.dot`,
 the result will be a 2D numpy array.  We can use `np.squeeze()` to remove redundant dimensions (in the case of single float,
  this will be reduced to a zero-dimension array). We can cast the array as a type `float` using `float()`.
'''
def compute_cost(A2, Y):
    cost = -np.sum( np.multiply(Y, np.log(A2)) )
    return np.squeeze(cost)


'''
**Question**: Implement the function `backward_propagation()`.

**Instructions**:
Backpropagation is usually the hardest (most mathematical) part in deep learning.
To help you, here again is the slide from the lecture on backpropagation. 
You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.  
'''
def backward_propagation(X, Y, cache, parameters):
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    m = X.shape[1]
    
    dZ2 = A2 - Y
    dW2 = (np.dot(dZ2, A1.T)) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


'''
**Question**: Implement the update rule. Use gradient descent. You have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).
'''
def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    updated_parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return updated_parameters





# Creat nn_model

'''
**Question**: Build your neural network model in `nn_model()`.
**Instructions**: The neural network model has to use the previous functions in the right order.
'''
def nn_model(X, Y, n_h, num_iterations, learning_rate):

    costs = []

    n_x, n_h, n_y = define_layer_size(X, Y, n_h)
    params = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        cache = forward_propagation(X, params)
        A2 = cache['A2']
        cost = compute_cost(A2, Y)
        grads = backward_propagation(X, Y, cache, params)
        params = update_parameters(params, grads, learning_rate)

        if i % 1000 == 0:
            costs.append(cost)

    return params, costs





# 4.5 Predictions

'''
**Question**: Use your model to predict by building predict().
Use forward propagation to predict results.
'''
def predict(params, X):
    caches = forward_propagation(X, params)
    return caches['A2'] > 0.5





# Build a model with a n_h-dimensional hidden layer
parameters, costs = nn_model(X, Y, 4,7000, 1.2)

plt.plot(costs)
plt.show()





'''
This code snippet is taken from the attached file in README.md
'''
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

print('NN model repot:')
predictions = predict(parameters, X)
accuracy = (
    np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)
) / float(Y.size) * 100

print("Accuracy: %d%%" % accuracy.item())