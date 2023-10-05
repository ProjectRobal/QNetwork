import numpy as np
import io

import config

from util import clip

from base.initializer import Initializer

class Neuron:
    def __init__(self,input_size:int,output_size:int,init:Initializer=None):
        '''
        input_size - a size of input weights
        output_size - a size of output weights

        state - a current state of a network
        '''
        # additional weight for past output, used for recurrsion
        # initial random weights
        if init is not None:
            self.input_weights=init.init(input_size)
            self.output_weights=init.init(output_size)
        
        self.state:float=0.0
        self.dot_product=config.DOT_PRODUCT
        # count in how many trials neuron has particpated
        self.trails=0

        # evaluation of neuron used for crossover and mutation 
        self.evaluation:float=0.0
        self.Q:float=0.0

    def fire(self,inputs:np.ndarray)->np.ndarray:

        self.state=self.dot_product(self.input_weights,inputs)

        return clip(self.output_weights*self.state)
    
    def Qvalue(self)->float:
        return self.Q

    def input_size(self)->int:
        return len(self.input_weights)
    
    def output_size(self)->int:
        return len(self.output_weights)

    def reset(self):
        self.state=0.0
        self.trails=0.0

    def reinitialize(self,init:Initializer):
        self.input_weights=init.init(self.input_size())
        self.output_weights=init.init(self.output_size())
    
    def Breedable(self)->bool:
        '''
            Whether the neuron is ready for breeding
        '''
        return self.trails==config.NUMBER_OF_TRIALS
    
    def Dumper(self,eval:float)->float:
        return 2*((1/(np.exp(-10*eval/(self.Q+0.00000000000001))+1)) - 0.5)
    
    def UpdateQ(self,eval:float):
        self.Q=clip(eval+config.LEARING_RATE*self.Q)

        self.trails+=1
        self.trails=clip(self.trails)

    def dump(self)->bytearray:
        '''
            Function used for neuron serialization,
            helpful for model saving
            Save inputs weights
            Save outputs weights
            Save trails numbers
        '''
        input_neurons=io.BytesIO()
        output_neurons=io.BytesIO()
        metadata=io.BytesIO()

        neuron_metadata=np.array([self.trails,self.Q],dtype=np.int32)

        np.save(metadata,neuron_metadata)
        np.save(input_neurons,self.input_weights)
        np.save(output_neurons,self.output_weights)

        metadata=metadata.getvalue()
        input_neurons=input_neurons.getvalue()
        output_neurons=output_neurons.getvalue()

        output=bytearray(metadata)
        output.extend(input_neurons)
        output.extend(output_neurons)

        return output

    def load(self,data:bytearray|io.BytesIO):
        '''
            Function used for neuron deserialization,
            helpful for model loading
        '''
        if type(data) is bytearray:
            inputs=io.BytesIO(data)
        else:
            inputs=data

        metadata=np.load(inputs)
        input_neurons=np.load(inputs)
        output_neurons=np.load(inputs)

        self.trails=metadata[0]
        self.Q=metadata[1]
        self.input_weights=input_neurons
        self.output_weights=output_neurons

    def __str__(self) -> str:
        return "Inputs: "+str(self.input_weights)+"\n"+"Outputs: "+str(self.output_weights)
