import numpy as np

from base.initializer import Initializer



class GaussInit(Initializer):
    def __init__(self,loc:float,scale:float) -> None:
        self.loc=loc
        self.scale=scale
        super().__init__()

    def init(self,size:int)->np.ndarray:
        return np.random.normal(self.loc,self.scale,size)