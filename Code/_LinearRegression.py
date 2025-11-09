import numpy as np
from sklearn.metrics import mean_squared_error
from mini_batch import mini_batch

class _LinearRegression(object):
    def __init__(self, model_type=None, gradient_descent_method = None, _param_getter = None, _gradient_descent_func = None):
        self.model_type = model_type
        self.model_params = None
        self.gradient_descent_method = gradient_descent_method
        self._param_getter = _param_getter
        self._gradient_descent_func = _gradient_descent_func
        self._num_points = None
        self._num_features = None
        self._features = None
        self._targets = None
    
    def fit(self, features, targets, **kwargs):
        """
        Computes model parameters by fitting features to targets.
        **kwargs are passed to the chosen gradient descent method.
        """
        self._features = features
        self._targets = targets.reshape(-1,1) #store targets as 2d-array
        self._num_features = np.shape(features)[1]
        self._num_points = len(targets)
        params, its, converged = self._param_getter(features, targets, **kwargs)
        self.model_params = params.T.flatten()
        return self.model_params, its, converged
    
    def predict(self, features, intercept = 0):
        """
        Calculate predicted y-values for a given feature matrix
        and previously calculated model parameters.
        """
        return features @ self.model_params.T + intercept

    # Gradient Descent methods:

    def _get_gd_method(self, gd_method):
        """
        All added gd methods need to be added here
        inorder to be callable by self.fit()
        """
        _gd_methods = {
        "simple": self._simple_gradient_descent,
        "momentum": self._momentum,
        "adagrad": self._adagrad,
        "RMSProp": self._rmsprop,
        "adam": self._adam,
        "simple_stochastic": self._simple_stochastic,
        "momentum_stochastic": self._momentum_stochastic,
        "adagrad_stochastic": self._adagrad_stochastic,
        "RMSProp_stochastic": self._rmsprop_stochastic,
        "adam_stochastic": self._adam_stochastic
        }
        return _gd_methods[gd_method]
    
    def _simple_gradient_descent(self, X, y, **kwargs):
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func_precomp())
        for feature matrix X and target values y by using simple gradient descent. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs:
        inital_theta = np.zeros()
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # Get kwargs, if not found give default value.
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1))) # initial guess
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        converged = True
        theta = theta.reshape(-1,1)
        i = 0
        grad = self._gradient_func_precomp(theta)
        while np.linalg.norm(grad) > precision:
            theta -= learning_rate*grad
            grad = self._gradient_func_precomp(theta)
            i += 1
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _momentum(self, X, y, **kwargs):
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func_precomp())
        for feature matrix X and target values y by using gradient descent with momentum. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs and their default values:
        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        momentum = 0.01

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        momentum = kwargs.get("momentum", 0.01)

        converged = True
        
        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        i = 0
        theta_old = theta
        old_change = np.zeros_like(theta)
        grad = self._gradient_func_precomp(theta)
        while np.linalg.norm(grad) > precision:
            # get change
            new_change = learning_rate*grad + momentum * old_change
            # update theta with change
            theta -= new_change
            # store change for next it
            old_change = new_change
            grad = self._gradient_func_precomp(theta)
            i += 1
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _adagrad(self, X, y, **kwargs):
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func_precomp())
        for feature matrix X and target values y by using AdaGrad. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs and their default values:
        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        num_stab_const = 1e-7

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables:
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        num_stab_const = kwargs.get("num_stab_const", 1e-7)

        converged = True

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        i = 0
        # accumulative sum of gradient element-wise squared
        grad_accum = np.zeros_like(theta)
        grad = self._gradient_func_precomp(theta)
        while np.linalg.norm(grad) > precision:
            # update theta
            grad_accum += np.square(grad)
            theta -= (learning_rate / ( num_stab_const + np.sqrt(grad_accum)) ) * grad
            # update change for next round
            grad = self._gradient_func_precomp(theta)
            i += 1
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _rmsprop(self, X, y, **kwargs): 
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func_precomp())
        for feature matrix X and target values y by using RMSProp. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs and their default values:

        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        decay_rate = 0.999
        num_stab_const = 1e-6

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables:
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        decay_rate = kwargs.get("decay_rate", 0.999)
        num_stab_const = kwargs.get("num_stab_const", 1e-6)

        converged = True

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        i = 0
        # decaying average of gradient element-wise squared
        grad_accum = np.zeros_like(theta)
        grad = self._gradient_func_precomp(theta)
        while np.linalg.norm(grad) > precision:
            grad_accum = decay_rate * grad_accum + (1-decay_rate)*np.square(grad)
            # update theta
            theta -= (learning_rate / ( num_stab_const + np.sqrt(grad_accum)) ) * grad
            # update change for next round
            grad = self._gradient_func_precomp(theta)
            i += 1
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _adam(self, X, y, **kwargs):
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func_precomp())
        for feature matrix X and target values y by using Adam. Runs until the gradient is less
        than the precision variable or until the number of iterations equal max_iter.

        **kwargs and their default values:

        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        first_moment = 0.9
        second_moment = 0.999
        num_stab_const = 1e-8

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables:
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        decay1 = kwargs.get("first_moment", 0.9)
        decay2 = kwargs.get("second_moment", 0.999)
        num_stab_const = kwargs.get("num_stab_const", 1e-8)

        converged = True

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError("X and y is not of right shape")
        i = 0
        # decaying average of gradient element-wise squared
        first_moment = np.zeros_like(theta)
        second_moment = np.zeros_like(theta)
        grad = self._gradient_func_precomp(theta)
        while np.linalg.norm(grad) > precision:
            i += 1
            first_moment = decay1 * first_moment + (1-decay1)*grad
            second_moment = decay2 * second_moment + (1-decay2)*np.square(grad)
            corrected_first = first_moment / (1 - decay1**i)
            corrected_second = second_moment / (1 - decay2**i)
            theta -= learning_rate * corrected_first / (np.sqrt(corrected_second) + num_stab_const)
            # update change for next round
            grad = self._gradient_func_precomp(theta)
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _simple_stochastic(self, X, y, **kwargs):
        """
        Find the model parameters theta that minimizes the gradient of the model object (self._gradient_func())
        for feature matrix X and target values y by using simple stochastic gradient descent. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs:
        inital_theta = np.zeros()
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        mbatch_size = 20

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables:
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        mbatch_size=kwargs.get("mini_batch_size", 20)

        converged = True

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        i = 0
        Xmb, ymb = mini_batch(X,y, mbatch_size)
        ymb = ymb.reshape(-1,1)
        grad = self._gradient_func(Xmb, ymb, theta)
        while np.linalg.norm(grad) > precision:
            theta -= learning_rate*grad
            i += 1
            if i >= max_iter:
                converged = False
                break
            Xmb, ymb = mini_batch(X,y, mbatch_size)
            ymb = ymb.reshape(-1,1)
            grad = self._gradient_func(Xmb, ymb, theta)
        return theta, i, converged

    def _momentum_stochastic(self, X, y, **kwargs):
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func())
        for feature matrix X and target values y by using gradient descent with momentum. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs and their default values:
        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        mini_batch_size = 20
        momentum = 0.01

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        mbatch_size=kwargs.get("mini_batch_size", 20)
        momentum = kwargs.get("momentum", 0.01)

        converged = True
        
        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        i = 0
        theta_old = theta
        old_change = np.zeros_like(theta)
        Xmb, ymb = mini_batch(X,y, mbatch_size)
        ymb = ymb.reshape(-1,1)
        grad = self._gradient_func(Xmb, ymb, theta)
        while np.linalg.norm(grad) > precision:
            # compute new change
            new_change = learning_rate*grad + momentum*old_change
            # update theta
            theta -= new_change
            # update old_change for next round
            old_change = new_change
            Xmb, ymb = mini_batch(X,y, mbatch_size)
            ymb = ymb.reshape(-1,1)
            grad = self._gradient_func(Xmb, ymb, theta)
            i += 1
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _adagrad_stochastic(self, X, y, **kwargs):
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func())
        for feature matrix X and target values y by using stochastic AdaGrad. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs and their default values:
        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        mini_batch_size = 20
        num_stab_const = 1e-7

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables:
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        mbatch_size = kwargs.get("mini_batch_size", 20)
        num_stab_const = kwargs.get("num_stab_const", 1e-7)

        converged = True

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        i = 0
        # accumulative sum of gradient element-wise squared
        grad_accum = np.zeros_like(theta)
        Xmb, ymb = mini_batch(X,y, mbatch_size)
        ymb= ymb.reshape(-1,1)
        grad = self._gradient_func(Xmb, ymb, theta)
        while np.linalg.norm(grad) > precision:
            # update theta
            grad_accum += np.square(grad)
            theta -= (learning_rate / ( num_stab_const + np.sqrt(grad_accum)) ) * grad
            # update change for next round
            Xmb, ymb = mini_batch(X,y, mbatch_size)
            ymb = ymb.reshape(-1,1)
            grad = self._gradient_func(Xmb, ymb, theta)
            i += 1
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _rmsprop_stochastic(self, X, y, **kwargs): 
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func())
        for feature matrix X and target values y by using RMSProp. Runs until the gradient is less
        than the precision variable or the number of iterations equal max_iter.

        **kwargs and their default values:

        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        mini_batch_size = 20
        decay_rate = 0.999
        num_stab_const = 1e-6

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables:
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        mbatch_size = kwargs.get("mini_batch_size", 20)
        decay_rate = kwargs.get("decay_rate", 0.999)
        num_stab_const = kwargs.get("num_stab_const", 1e-6)

        converged = True

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        i = 0
        # decaying average of gradient element-wise squared
        grad_accum = np.zeros_like(theta)
        Xmb, ymb = mini_batch(X,y, mbatch_size)
        ymb = ymb.reshape(-1,1)
        grad = self._gradient_func(Xmb, ymb, theta)
        while np.linalg.norm(grad) > precision:
            grad_accum = decay_rate * grad_accum + (1-decay_rate)*np.square(grad)
            # update theta
            theta -= (learning_rate / ( num_stab_const + np.sqrt(grad_accum)) ) * grad
            # update change for next round
            Xmb, ymb = mini_batch(X,y, mbatch_size)
            ymb = ymb.reshape(-1,1)
            grad = self._gradient_func(Xmb, ymb, theta)
            i += 1
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged

    def _adam_stochastic(self, X, y, **kwargs):
        """
        Find the model parameters theta for that minimizes the gradient of the model object (self._gradient_func())
        for feature matrix X and target values y by using Adam. Runs until the gradient is less
        than the precision variable or until the number of iterations equal max_iter.

        **kwargs and their default values:

        inital_theta = np.zeros((num_features,1))
        learning_rate = 0.01
        precision = 1e-8
        max_iter = 10000

        first_moment = 0.9
        second_moment = 0.999
        num_stab_const = 0.01

        Returns the computed theta, the number of iterations completed i, and the bool converged.
        """
        # general gradient descent variables:
        theta = kwargs.get("initial_theta", np.zeros((self._num_features,1)))
        learning_rate = kwargs.get("learning_rate",0.01)
        precision = kwargs.get("precision", 1e-8)
        max_iter = kwargs.get("max_iter", 10000)

        # algorithm specific variables
        mbatch_size = kwargs.get("mini_batch_size", 20)
        decay1 = kwargs.get("first_moment", 0.9)
        decay2 = kwargs.get("second_moment", 0.999)
        num_stab_const = kwargs.get("num_stab_const", 1e-8)

        converged = True

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError("X and y is not of right shape")
        i = 0
        # decaying average of gradient element-wise squared
        first_moment = np.zeros_like(theta)
        second_moment = np.zeros_like(theta)
        Xmb, ymb = mini_batch(X,y, mbatch_size)
        ymb = ymb.reshape(-1,1)
        grad = self._gradient_func(Xmb, ymb, theta)
        while np.linalg.norm(grad) > precision:
            i += 1
            first_moment = decay1 * first_moment + (1-decay1)*grad
            second_moment = decay2 * second_moment + (1-decay2)*np.square(grad)
            corrected_first = first_moment / (1 - decay1**i)
            corrected_second = second_moment / (1 - decay2**i)
            theta -= learning_rate * corrected_first / (np.sqrt(corrected_second) + num_stab_const)
            # update change for next round
            Xmb, ymb = mini_batch(X,y, mbatch_size)
            ymb = ymb.reshape(-1,1)
            grad = self._gradient_func(Xmb, ymb, theta)
            if i >= max_iter:
                converged = False
                break
        return theta, i, converged
