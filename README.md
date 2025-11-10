# Project 2, FYS-STK4155 

#### This repository is for Project 2 on Neural Networks in the course **FYS-STK4155 - Applied Data Analysis and Machine Learning**, as taught at the **University of Oslo** during the autumn semester of 2025.

In this project we implemented a Feedforward Neural Network from scratch and tested it on several regression and classification problems.

## Authors

- Simen Lund Wærstad [@Waerstad](https://github.com/Waerstad)
- John-Magnus Johnsen [@johnmagj](https://github.com/johnmagj)

## Requirements

All required python packages are listed in the file `requirements.txt`. These packaged can be installed using the following terminal command: ``pip install -r requirements.txt``

## The Code
All the code used for the project is found in the folder `Code/`. The python file `NeuralNetwork.py` contains a class that implements a flexible neural network. Similarly, `Scheduler.py` contains the a class that implements gradient descent schedulers used by the `NeuralNetwork` class. All additional cost and activation functions are listed in the file `functions.py`. Lastly, the file `plotting.py` contains some configurations to make matplotlib plots look nice.

The jupyter-notebooks contained in `Code/` were used to carry out any calculations used in the paper.

## Documentation

### NeuralNetwork
The `NeuralNetwork` class:
```
NeuralNetwork(n_inputs, n_outputs, n_nodes_hidden_layers, activation_funcs,
              activation_derivs, scheduler, cost_func, cost_func_der,
              reg_param = 0, classifier_mode = False, regularization = None,
              seed = None
              )
```
A class implementing a feed-forward neural network with back propagation.

`n_inputs:`               The number of nodes in the input layer.

`n_outputs`:              The number of nodes in the output layer.

`n_nodes_hidden_layers`:  List of the number of nodes in the 
                        hidden layers.

`activation_funcs`:       List containing the activation funcs for 
                        each hidden layer and the output layer.
                        The activation for the output layer is
                        assumed to be the last entry.

`activation_derivs`:      List containing the derivatives of the
                        activation functions.

`scheduler`:              The Scheduler used for gradient descent.
                        Must be a scheduler object.

`cost_func`:              The cost func to be used by backpropagation.
                        Is not used.

`cost_func_der`:          The derivative of the cost func to be used by
                        backpropagation. Is not used if
                        classifier_mode = True.

`reg_param`:              The hyperparameter used as the coefficient
                        for the regularization term.

`classifier_mode`:        Bool that enables classifier mode. If
                        enabled cost_func and cost_func_der is
                        ignored and the cost function is always
                        set to cross-entropy. This mode requires
                        the output activation function to be
                        softmax.

`regularization`:         String that decides what type of regularization
                        to use. "L1" gives L_1 regularization. "L2" gives
                        L_2 regularization. Anything else gives no
                        regularization. Uses reg_param as the
                        regularization hyperparameter.

`seed`:                   Sets a seed that is used for selecting random
                        batches and initalizing weights.

It has the following user-facing methods:

```
predict(input_data)
```
Performs a feed forward pass on input_data and returns the resulting prediction.

```
train(training_features, training_targets, batch_size, num_epochs)
```
Trains the neural network using training_features and training_targets.
For gradient descent the training_features are divided into batches of
size batch_size. The number of epochs is set by num_epochs. Returns nothing.

```
reset_weights()
```
Resets the weights be re-initializing them.

### Scheduler
The `Scheduler` class is an abstract class used as a template to implement other 
It has the attribute `learning_rate` which sets the learning rate for the scheduler.
It has two methods. `change(gradient)` computes the gradient based on `gradient`.
The method `reset` resets the scheduler, setting any saved data (such as momentum)
to its initial value.

There are three different Scheduler objects:

```
Simple(learning_rate)
```
which does simple gradient descent.

```
RMSProp(learning_rate, decay_rate=0.999, num_stab_const=1e-6)
```
which performs gradient descent with the RMSProp optimizer.

```
Adam(learning_rate, first_moment = 0.9, second_moment = 0.999, num_stab_const = 1e-8)
```
which performs gradient descent with the Adam optimizer.





