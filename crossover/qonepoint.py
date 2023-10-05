import numpy as np

import neuron
from base.crossover import Crossover

class QOnePoint(Crossover):
    '''
        A base class for implementing crossover method for neurons and networks.
        So each function has NotImplementedError.
        It takes into account Q value crossover.
    '''
    @staticmethod
    def cross_numpy(x1:np.array,x2:np.array)->np.array:
        output=np.ndarray(len(x1),np.float32)

        output[0:int(len(x1)/2)]=x2[0:int(len(x1)/2)]
        output[(int(len(x1)/2)):len(x1)]=x1[(int(len(x1)/2)):len(x1)]

        return output
        

    @staticmethod
    def CrossNeurons(neuron1:neuron.Neuron,neuron2:neuron.Neuron)->neuron.Neuron:
        out=neuron.Neuron(neuron1.input_size(),neuron1.output_size())

        out.input_weights=QOnePoint.cross_numpy(neuron1.input_weights,neuron2.input_weights)
        out.output_weights=QOnePoint.cross_numpy(neuron1.output_weights,neuron2.output_weights)

        out.Q=(neuron1.Q+neuron2.Q)/2.0

        return out