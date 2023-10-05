from base.mutation import Mutation

import numpy as np

import neuron

class GaussMutaion(Mutation):
    '''
        A base class for implementing mutation method for neurons and networks.
        So each function has NotImplementedError.
    '''

    @staticmethod
    def MutateNeuron(neuron:neuron.Neuron)->neuron.Neuron:

        #if np.random.random()<0.1:

        neuron.input_weights+=np.random.normal(0,0.01,len(neuron.input_weights))
        neuron.output_weights+=np.random.normal(0,0.01,len(neuron.output_weights))

        return neuron