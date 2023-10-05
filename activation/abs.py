import numpy as np

from base.activation import Activation

class ABS(Activation):
    @staticmethod
    def activate(x: float) -> float:
        if x>=0.0:
            return x
        return -x
    