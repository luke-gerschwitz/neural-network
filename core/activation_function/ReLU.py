from core.activation_function.base_activation import BaseActivation

class ReLU(BaseActivation):
    name = "ReLU"

    def perform_activation(self, z):
        print(f"Here {z}")
        if z < 0:
            return 0
        return z