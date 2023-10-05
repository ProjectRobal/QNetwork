'''
    A file that store base class for performing mutation on neurons and networks.

'''

from typing import Any
import neuron

class Mutation:
    '''
        A base class for implementing mutation method for neurons and networks.
        So each function has NotImplementedError.
    '''

    @staticmethod
    def MutateNeuron(neuron:neuron.Neuron)->neuron.Neuron:
        raise NotImplementedError()
    
    def __new__(cls, neuron:neuron.Neuron) -> neuron.Neuron:
        return cls.MutateNeuron(neuron)