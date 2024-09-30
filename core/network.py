import numpy as np
from core.neuron.input_neuron import InputNeuron
from core.neuron.hidden_neuron import HiddenNeuron
from core import operations

class NeuralNetwork():
    input_layer: np.ndarray = None
    hidden_layer: np.ndarray = None
    output_layer: np.ndarray = None

    def __init__(self,
                 input_layer_size,
                 output_layer_size,
                 hidden_layer_size, 
                 no_hidden_layers=1,
                 random_init=True
        ):
        self.input_layer = np.full(shape=input_layer_size, fill_value=InputNeuron(), dtype=np.dtype(InputNeuron))
        self.hidden_layer = np.full(shape=hidden_layer_size, fill_value=HiddenNeuron(), dtype=np.dtype(HiddenNeuron))
        self.output_layer = np.full(shape=output_layer_size, fill_value=HiddenNeuron(), dtype=np.dtype(HiddenNeuron))

        if random_init:
            self.input_layer.randomly_initialise_weights()
    
    def invoke(self, inputs: np.ndarray):
        return None