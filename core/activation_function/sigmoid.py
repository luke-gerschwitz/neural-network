from core.activation_function.base_activation import BaseActivation
from core import operations

class Sigmoid(BaseActivation):
    name = "Sigmoid"

    def perform_activation(self, z):
        return operations.sigmoid(z)