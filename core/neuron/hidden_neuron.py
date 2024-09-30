import numpy as np
from core.neuron.unit import Unit
from core.activation_function.base_activation import BaseActivation

class HiddenNeuron(Unit):

    weights: np.ndarray = None
    bias: float = None
    activation_function: BaseActivation = None

    def __init__(self, weights, bias, activation_function: BaseActivation = None):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def calculate_activation(self, x):
        if self.activation_function is not None:
            return self.activation_function.perform_activation(x)
        return x

    def calculate_output(self, inputs: np.ndarray):
        weighted_input = np.dot(inputs, self.weights)
        return self.calculate_activation(np.sum(weighted_input) + self.bias)
    
    def randomly_initialise_weights(self):
        self.weights = np.random.rand()