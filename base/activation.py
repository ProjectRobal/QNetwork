'''
    A base class for activation function for neurons.
'''

class Activation:
    @staticmethod
    def activate(x:float)->float:
        raise NotImplementedError()        
    
    def __new__(cls, x:float) -> float:
        return cls.activate(x)