import numpy as np
import config

def clip(x:float|np.ndarray)->float|np.ndarray:
    return np.clip(x,config.MIN_VALUE_NUMBER,config.MAX_VALUE_NUMBER)