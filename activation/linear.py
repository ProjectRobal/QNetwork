from base.activation import Activation

class Linear(Activation):
    @staticmethod
    def activate(x: float) -> float:
        return x