'''
    A file that store base class for dot product calculation for neurons.
'''

from typing import Any
import numpy as np

class Product:
    @staticmethod
    def compute(x1:np.array,x2:np.array)->float:
        raise NotImplementedError()
    
    def __new__(cls,x1:np.array,x2:np.array)->float:
        return cls.compute(x1,x2)
    