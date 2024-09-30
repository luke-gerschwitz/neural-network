from abc import ABC, abstractmethod

class BaseActivation(ABC):
    
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def perform_activation(self, z):
        pass
