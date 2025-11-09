import autograd.numpy as np

"""
Activation functions and their derivatives
"""

def sigmoid(x):
    #try
    return 1.0 / (1 + np.exp(-x))
    #except FloatingPointError:
    #   return np.where(x > np.zeros(x.shape), np.ones(x.shape), np.zeros(x.shape))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def ReLU(x):
    return np.where(x > 0, x, 0)

def ReLU_der(x):
    return np.where(x > 0, 1, 0)

def leaky_ReLU(x):
    return np.where(x > 0, x, 0.01*x)

def leaky_ReLU_der(x):
    return np.where(x > 0, 1, 0.01)

def id_func(x):
    return x

def id_func_der(x):
    return np.ones_like(x)

"""
cost functions
"""

def mse(predict,target):
    return np.mean((predict-target)**2)

def mse_der(predict, target):
    n = np.prod(np.shape(predict))
    return 2/n * (predict-target)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-10)


