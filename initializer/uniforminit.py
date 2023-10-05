import numpy as np

from base.initializer import Initializer


class UniformInit(Initializer):
    def __init__(self) -> None:
        super().__init__()

    def init(self,size:int)->np.ndarray:
        return np.random.random(size)