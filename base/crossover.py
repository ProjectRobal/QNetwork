'''
    A file that store base class for performing crossover between nuerons and networks schematics.

'''

from typing import Any
import neuron

class Crossover:
    '''
        A base class for implementing crossover method for neurons and networks.
        So each function has NotImplementedError.
    '''

    @staticmethod
    def CrossNeurons(neuron1:neuron.Neuron,neuron2:neuron.Neuron)->neuron.Neuron:
        raise NotImplementedError()
    
    def __new__(cls, neuron1:neuron.Neuron,neuron2:neuron.Neuron) -> neuron.Neuron:
        return cls.CrossNeurons(neuron1,neuron2)