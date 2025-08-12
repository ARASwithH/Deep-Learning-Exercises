import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)




# L-layer Neural Network 
'''
**Exercise**: Implement initialization for an L-layer Neural Network. 

**Instructions**:
- The model's structure is *[LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
- Use random initialization for the weight matrices. Use `np.random.randn(shape) * 0.01`.
- Use zeros initialization for the biases. Use `np.zeros(shape)`.
- We will store $n^{[l]}$, the number of units in different layers, in a variable `layer_dims`. For example, the `layer_dims` for the "Planar Data classification model" from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. This means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers! 
- Here is the implementation for $L=1$ (one layer neural network). It should inspire you to implement the general case (L-layer neural network).
```python
    if L == 1:
        parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
        parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
```
'''

def initialization_params(dims):
    params = {}
    n = len(dims)

    for i in range(n-1):
        params["W" + str(i+1)] = np.random.randn(dims[i+1], dims[i]) * 0.01
        params["b" + str(i+1)] = np.zeros((dims[i+1], 1))

    return params




# Forward propagation module

# Linear Forward 
'''
**Exercise**: Build the linear part of forward propagation.

**Reminder**:
The mathematical representation of this unit is $Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$.
 You may also find `np.dot()` useful. If your dimensions don't match, printing `W.shape` may help.
'''

def linear_forward_prop(W, A, b):
    return np.dot(W, A) + b , (W, A, b)



# Linear-Activation Forward
'''
**Exercise**: Implement the forward propagation of the *LINEAR->ACTIVATION* layer.
 Mathematical relation is: $A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} +b^{[l]})$ where the activation "g" can be sigmoid() or relu().
 Use linear_forward() and the correct activation function.
'''

def linear_activation_forward_prop(W, A, b, g):
    
    Z, linear_cache = linear_forward_prop(W, A, b)
    
    if g == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    
    if g == 'relu':
        A, activation_cache = relu(Z)

    return A , (linear_cache, activation_cache)




# L-Layer Model 
'''
**Exercise**: Implement the forward propagation of the above model.

**Instruction**: In the code below, the variable `AL` will denote $A^{[L]} = \sigma(Z^{[L]}) = \sigma(W^{[L]} A^{[L-1]} + b^{[L]})$.
(This is sometimes also called `Yhat`, i.e., this is $\hat{Y}$.) 

**Tips**:
- Use the functions you had previously written 
- Use a for loop to replicate [LINEAR->RELU] (L-1) times
- Don't forget to keep track of the caches in the "caches" list. To add a new value `c` to a `list`, you can use `list.append(c)`.
'''

def L_model_forward_prop(X, params):

    caches = []
    L = int(len(params) / 2)
    A = X
    for i in range(L-1):
        A_prev = A
        W = params['W'+str(i+1)]
        b = params['b'+str(i+1)]
        g = 'relu'
        A, cache = linear_activation_forward_prop(W, A_prev, b, g)
        caches.append(cache)


    W = params['W'+str(L)]
    b = params['b'+str(L)]
    g = 'sigmoid'
    AL, cache = linear_activation_forward_prop(W, A, b, g)
    caches.append(cache)

    return AL, caches
