from base.activation import Activation

class Relu(Activation):
    @staticmethod
    def activate(x: float) -> float:
        if x>=0.0:
            return x
        
        return 0.0