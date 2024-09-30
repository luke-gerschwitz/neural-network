from core.activation_function.base_activation import BaseActivation
from core import operations

class Tanh(BaseActivation):
    name = "Tanh"

    def perform_activation(self, z):
        return operations.tanh(z)