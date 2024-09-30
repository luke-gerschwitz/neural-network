from abc import ABC, abstractmethod

class Unit(ABC):

    @abstractmethod
    def calculate_output(self):
        pass
