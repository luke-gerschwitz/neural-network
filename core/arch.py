import numpy as np
from core.activation_function.base_activation import BaseActivation

class Arch():
    inputs: np.ndarray = None  # Vector of inputs (x1, x2, xn)
    weights: np.ndarray = None  # 3D array of weights (Each column is a neuron's weights)
    bias: np.ndarray = None  # Vector of biases (b1, b2, bn)
    activation_function: BaseActivation = None  # Activation function (e.g. sigmoid, ReLU)

    def __init__(self,
                 inputs: np.ndarray=None,
                 weights: np.ndarray=None,
                 bias: np.ndarray=None,
                 activation_function: BaseActivation = None
    ):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def invoke(self):
        t_weights = np.transpose(self.weights)
        weighted_sum = np.dot(t_weights, self.inputs) + self.bias
        return np.array([self.activation_function.perform_activation(x) for x in weighted_sum])
