from core.neuron.unit import Unit

class InputNeuron(Unit):

    def __init__(self, input=None):
        self.input = input

    def calculate_output(self):
        return self.input