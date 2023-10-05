import numpy as np

from base.activation import Activation

class Sigmoid(Activation):
    @staticmethod
    def activate(x: float) -> float:
        return np.exp(x)/(1.0+np.exp(x))