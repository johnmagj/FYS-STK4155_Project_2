import numpy as np

class Scheduler:
    "Class used to implement different schedulers with a common structure and interface."
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def change(self, gradient):
        """
        Returns the change computed by the optimizer given the gradient.
        """
        raise NotImplementedError
    
    def reset(self):
        """
        Resets the Scheduler.
        """
        pass

class Simple(Scheduler):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def change(self, gradient):
        """
        Returns the change computed by the optimizer given the gradient.
        """
        return self.learning_rate * gradient
    
class RMSProp(Scheduler):
    """
    RMSProp scheduler
    """
    def __init__(self, learning_rate, decay_rate = 0.999, num_stab_const = 1e-6):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.num_stab_const = num_stab_const
        self.grad_accum = 0
        
    def change(self, gradient):
        """
        Returns the change computed by the optimizer given the gradient.
        """
        #compute gadient accumulation
        self.grad_accum = self.decay_rate * self.grad_accum + (1-self.decay_rate)*np.square(gradient)
        # return change in gradient
        return (self.learning_rate / (self.num_stab_const + np.sqrt(self.grad_accum)) ) * gradient
    
    def reset(self):
        """
        Resets Scheduler.
        """
        self.grad_accum = 0

class Adam(Scheduler):
    """
    Adam scheduler
    """
    def __init__(self, learning_rate, first_moment = 0.9, second_moment = 0.999, num_stab_const = 1e-8):
        super().__init__(learning_rate)
        self.decay1st = first_moment
        self.decay2nd = second_moment
        self.first_moment = 0
        self.second_moment = 0
        self.num_epochs = 1
        self.num_stab_const = num_stab_const
        

    def change(self, gradient):
        """
        Returns the change computed by the optimizer given the gradient.
        """
        self.first_moment = self.decay1st * self.first_moment + (1-self.decay1st)*gradient
        self.second_moment = self.decay2nd * self.second_moment + (1-self.decay2nd)*np.square(gradient)
        corrected_first = self.first_moment / (1 - self.decay1st**self.num_epochs)
        corrected_second = self.second_moment / (1 - self.decay2nd**self.num_epochs)
        self.num_epochs += 1
        return self.learning_rate * corrected_first / (np.sqrt(corrected_second) + self.num_stab_const)
    
    def reset(self):
        """
        Resets scheduler.
        """
        self.first_moment = 0
        self.second_moment = 0
        self.num_epochs = 1

        


