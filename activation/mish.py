import numpy as np

from base.activation import Activation

class Mish(Activation):
    @staticmethod
    def activate(x: float) -> float:
        return x*np.tanh(np.log(1+np.exp(x)))