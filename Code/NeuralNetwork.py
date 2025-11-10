import autograd.numpy as np
import Scheduler
from copy import deepcopy

class NeuralNetwork(object):
    """
    A class implementing a feed-forward neural network with back propagation.

    n_inputs:               The number of nodes in the input layer.

    n_outputs:              The number of nodes in the output layer.

    n_nodes_hidden_layers:  List of the number of nodes in the 
                            hidden layers.

    activation_funcs:       List containing the activation funcs for 
                            each hidden layer and the output layer.
                            The activation for the output layer is
                            assumed to be the last entry.

    activation_derivs:      List containing the derivatives of the
                            activation functions.

    scheduler:              The Scheduler used for gradient descent.
                            Must be a scheduler object.

    cost_func:              The cost func to be used by backpropagation.
                            Is not used.

    cost_func_der:          The derivative of the cost func to be used by
                            backpropagation. Is not used if
                            classifier_mode = True.

    reg_param:              The hyperparameter used as the coefficient
                            for the regularization term.

    classifier_mode:        Bool that enables classifier mode. If
                            enabled cost_func and cost_func_der is
                            ignored and the cost function is always
                            set to cross-entropy. This mode requires
                            the output activation function to be
                            softmax.

    regularization:         String that decides what type of regularization
                            to use. "L1" gives L_1 regularization. "L2" gives
                            L_2 regularization. Anything else gives no
                            regularization. Uses reg_param as the
                            regularization hyperparameter.

    seed:                   Sets a seed that is used for selecting random
                            batches and initalizing weights.


    """

    def __init__(
            self, 
            n_inputs,
            n_outputs,
            n_nodes_hidden_layers,
            activation_funcs,
            activation_derivs,
            scheduler,
            cost_func,
            cost_func_der,
            reg_param = 0,
            classifier_mode = False,
            regularization = None,
            seed = None
    ):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._n_hidden_layers = len(n_nodes_hidden_layers)
        self._n_nodes_hidden_layers = n_nodes_hidden_layers
        self._activation_funcs = activation_funcs
        self._activation_derivs = activation_derivs
        self._cost_func = cost_func
        self._cost_func_der = cost_func_der
        self._seed = seed
        self._reg_param = reg_param
        self._regularization = regularization
        self._scheduler = scheduler
        self._weights_biases = self._init_weights()
        self._classifier_mode = classifier_mode
        

        self._layer_inputs = [] # usually denoted by 'a'
        self._layer_values = [] # Usually denoted by 'z'
        self._training_data = []

        self._init_schedulers()

    def _init_weights(self):
        """
        initalizes weights.
        """
        if self._seed != None:
            np.random.seed(self._seed)

        weights_biases = [(None, None)] * (self._n_hidden_layers + 1)
        n_layer_inputs = self._n_inputs
        for i, n_nodes in enumerate(self._n_nodes_hidden_layers):
            weights = np.random.randn(n_layer_inputs, n_nodes)
            biases = np.random.randn(n_nodes,1)
            weights_biases[i] = (weights, biases)
            # set number of inputs for next layer as the number of nodes of current layer
            n_layer_inputs = n_nodes
        # compute weights and biases for output layer
        weights = np.random.randn(n_layer_inputs, self._n_outputs)
        biases = np.zeros((self._n_outputs, 1)) + 0.01
        weights_biases[-1] = (weights, biases)
        return weights_biases
    
    def _init_schedulers(self):
        """
        Creates and stores a scheduler for each weight
        and bias matrix.
        """
        self._weights_schedulers = []
        self._biases_schedulers = []
        for i in self._weights_biases:
            scheduler_weight = deepcopy(self._scheduler)
            scheduler_bias = deepcopy(self._scheduler)
            self._weights_schedulers.append(scheduler_weight)
            self._biases_schedulers.append(scheduler_bias)

    def _reset_schedulers(self):
        """
        Resets all scheduler objects.
        """
        for weight_scheduler in self._weights_schedulers:
            weight_scheduler.reset()
        for bias_scheduler in self._biases_schedulers:
            bias_scheduler.reset()

    
    def _reset_inputs_and_activations(self):
        """
        Resets the saved layer inputs and values (weight*input + bias) used for back propogation.
        """
        self._layer_values = []
        self._layer_inputs = []
        
    def _feed_forward(self, input_data):
        """
        Feed forward pass that saves node values and activations of each layer
        for backpropagation used for training.
        """
        self._reset_inputs_and_activations()
        nodes_inputs = input_data
        for (weights, biases), activation_func in zip(self._weights_biases, self._activation_funcs):
            self._layer_inputs.append(nodes_inputs)
            nodes_values = nodes_inputs @ weights + biases.T
            nodes_inputs = activation_func(nodes_values)
            self._layer_values.append(nodes_values)
        return nodes_inputs
    
    def _feed_forward_out(self, input_data):
        """
        Feed forward pass that does NOT save node values and activations of each layer
        for backpropagation.
        """
        nodes_inputs = input_data
        for (weights, biases), activation_func in zip(self._weights_biases, self._activation_funcs):
            nodes_values = nodes_inputs @ weights + biases.T
            nodes_inputs = activation_func(nodes_values)
        return nodes_inputs
    
    def _backpropagation(self,
                         target,
                         prediction):
        """
        Does a backpropagation pass for the given input and 
        """
        cost_der = self._cost_func_der
        activation_ders = self._activation_derivs

        layers = self._weights_biases
        layer_inputs = self._layer_inputs
        layer_vals = self._layer_values
        layer_grads = [() for layer in layers]
    
        # calculate layer error and gradients for output layer.
        if self._classifier_mode:
            error_current = prediction - target
        else:
            act_der_val  = activation_ders[-1](layer_vals[-1])
            error_current = cost_der(prediction, target) * act_der_val

        weight_grad = layer_inputs[-1].T @ error_current
        bias_grad = np.sum(error_current, axis=0)[:,np.newaxis]
        layer_grads[-1] = (weight_grad, bias_grad)

        error_prev = error_current
        # We loop over the hidden layers, from the last to the first
        for i in reversed(range(len(layers)-1)):
            act_der = activation_ders[i]
            layer_val = layer_vals[i]
            layer_input = layer_inputs[i]
            weight_prev, _ = layers[i+1]
            error_current = error_prev @ weight_prev.T * act_der(layer_val)
            weight_grad = layer_input.T @ error_current
            bias_grad = np.sum(error_current, axis=0)[:,np.newaxis]
            layer_grads[i] = (weight_grad, bias_grad)
            
            error_prev = error_current
            
        return layer_grads

    def predict(self, input_data):
        """
        Performs a feed forward pass on input_data
        and returns the resulting prediction.
        """
        return self._feed_forward_out(input_data)

    
    def train(self, training_features, training_targets, batch_size, num_epochs):
        """
        Trains the neural network using training_features and training_targets.
        For gradient descent the training_features are divided into batches of
        size batch_size. The number of epochs is set by num_epochs. Returns nothing.
        """

        if self._regularization == "L1":
            def reg(weight):
                return (self._reg_param/weight.size) * np.sum(np.abs(weight))
        elif self._regularization == "L2":
            def reg(weight):
                return self._reg_param/weight.size * np.sum(np.square(weight))
        else:
            def reg(weight):
                return 0
        
        num_batches = len(training_features) // batch_size
        data_indices = np.arange(len(training_features))

        for i_epoch in range(num_epochs):
            for i_batch in range(num_batches):
                batch_indices = np.random.choice(data_indices, size=batch_size, replace=False)
                batch_data = training_features[batch_indices]
                batch_targets = training_targets[batch_indices]

                prediction = self._feed_forward(batch_data)
                layer_grads = self._backpropagation(batch_targets, prediction)
                for i, (grad_weight, grad_bias) in enumerate(layer_grads):
                    weight, bias = self._weights_biases[i]
                    grad_weight += reg(weight)
                    grad_bias += reg(bias)

                    change_weight = self._weights_schedulers[i].change(grad_weight)
                    change_bias = self._biases_schedulers[i].change(grad_bias)

                    self._weights_biases[i] = (weight - change_weight, bias - change_bias)
            self._reset_schedulers()

    def reset_weights(self):
        """
        Resets the weights be re-initializing them.
        """
        self._init_weights()



