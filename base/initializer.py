'''

 Abstract class for weight initialization.


'''

import numpy as np

class Initializer:


    def init(self,size:int)->np.ndarray:
        raise NotImplementedError()