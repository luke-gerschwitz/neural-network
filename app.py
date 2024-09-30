import numpy as np
from core.activation_function.ReLU import ReLU
from core.neuron import hidden_neuron, input_neuron
from core.network import NeuralNetwork
from core.arch import Arch

def xor():
    x1 = input_neuron.InputNeuron(1)
    x2 = input_neuron.InputNeuron(0)

    h1 = hidden_neuron.HiddenNeuron(
        weights=np.array([1, 1]),
        bias = 0,
        activation_function=ReLU()
    )

    h2 = hidden_neuron.HiddenNeuron(
        weights=np.array([1, 1]),
        bias = -1,
        activation_function=ReLU()
    )

    y1 = hidden_neuron.HiddenNeuron(
        weights=np.array([1, -2]),
        bias = 0,
    )

    h1_output = h1.calculate_output(inputs=np.array([
                    x1.calculate_output(),
                    x2.calculate_output()
                ]))
    h2_output = h2.calculate_output(inputs=np.array([
                    x1.calculate_output(),
                    x2.calculate_output()
                ]))
    
    print(y1.calculate_output(np.array([h1_output, h2_output])))

def feedforward():
    net = NeuralNetwork(
        input_layer_size=3,
        output_layer_size=10,
        hidden_layer_size=10
    )

    print(type(net.input_layer[0]))

def matrix():
    inputs = np.array([0.5, 1.0, -1.5])
    weights = np.array([[0.2, 0.4],
                       [-0.3, 0.5], 
                       [0.1, -0.2]])
    bias = np.array([0.1, -0.1])

    neural_net = Arch(inputs, weights, bias, ReLU())

    print(neural_net.invoke())

if __name__ == "__main__":
    # xor()
    # feedforward()
    matrix()