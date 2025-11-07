import numpy as np

"""
Activation functions and their derivatives
"""

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_der(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

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
    return x*0+1

"""
cost functions
"""

def mse(predict,target):
    return np.mean((predict-target)**2)

def mse_der(predict, target):
    return 2*np.mean(predict-target)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / np.sum(e_x, axis=1)[:, np.newaxis]


