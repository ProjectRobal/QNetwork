import io
import numpy as np
import pickle as pkl
from base.activation import Activation
from base.initializer import Initializer

from activation.linear import Linear
from BreedStrategy import BreedStrategy
from initializer.uniforminit import UniformInit

import block

from util import clip

class Layer:
    '''
        A class that stores blocks, it is recursive layer
    '''
    def __init__(self,input_size:int,output_size:int,block_number:int,block_nueron_number:int=64,block_population_size:int=512,init:Initializer=UniformInit(),breed_strategy=BreedStrategy()) -> None:


        self.breed_strategy=breed_strategy

        # it gets outputs from last step
        self.input_size=input_size
        self.output_size=output_size

        # output activation function
        self.activation_fun:list[Activation]=[Linear]*self.output_size
        
        self.blocks:list[block.Block]=[]

        self.init_blocks(block_number,block_nueron_number,block_population_size,init)
    
    def init_blocks(self,block_number:int,block_nueron_number:int,block_population_size:int,init:Initializer):
        
        for n in range(block_number):
            _block=block.Block(self.input_size,self.output_size,block_nueron_number,block_population_size,self.breed_strategy)
            _block.setInitializer(init)
            _block.createPopulation()
            self.blocks.append(_block)

    def setActivationFun(self,activ_fun:list[Activation]):
        if len(activ_fun)!=self.output_size:
            raise ValueError("Activation function list doesn't have required size")

        self.activation_fun=activ_fun

    def reset(self):
        for block in self.blocks:
            block.clearPopulation()
            block.clearPopulation()

    def pickNBatch(self):
        for block in self.blocks:
            block.pickBatch()

    def fire(self,_inputs:np.ndarray)->np.ndarray:
        outputs:np.ndarray=np.zeros(self.output_size,dtype=np.float32)

        for block in self.blocks:
            block.pickBatch()
            outputs+=block.fire(_inputs)#/len(self.blocks)

        for n,activ in enumerate(self.activation_fun):
            outputs[n]=clip(activ(outputs[n]))

        return outputs
    
    def step(self,_inputs:np.ndarray)->np.ndarray:
        outputs:np.ndarray=np.zeros(self.output_size,dtype=np.float32)

        for block in self.blocks:
            #block.pickBatch()
            outputs+=block.fire(_inputs)#/len(self.blocks)

        for n,activ in enumerate(self.activation_fun):
            outputs[n]=clip(activ(outputs[n]))

        return outputs


    def evalute(self,eval:float):
        '''
            A function that evenly pass evaluation to blocks
        '''
        eval=eval/len(self.blocks)

        for block in self.blocks:
            block.Evaluate(eval)

    def changeBestRatioPopulation(self,depsilon:float):
        for block in self.blocks:
            block.updateEpsilon(depsilon)

    def getBestRatioPopulation(self)->float:
        return self.blocks[0].getEpsilon()        

    def mate(self):
        for block in self.blocks:
            if block.ReadyForMating():
                block.Mating()

    def save(self,memory:io.BufferedIOBase):
        '''
            Save each blocks
            Activation functions list
            Input size
            Output size
            Number of blocks
            Last output

            Every block will be saved in individual file
        '''

        # layer type 0x01 is recurrent type

        np.save(memory,np.array([0x01],dtype=np.int16))

        metadata=np.array([self.input_size,self.output_size,len(self.blocks)],dtype=np.int32)

        np.save(memory,metadata)

        for block in self.blocks:
            block.save(memory)

        pkl.dump(self.activation_fun,memory)
 
    def load(self,data:io.RawIOBase):
        
        metadata=np.load(data)

        self.input_size=metadata[0]
        self.output_size=metadata[1]

        self.blocks.clear()

        for i in range(metadata[2]):
            _block=block.Block(0,0,0,0)
            _block.load(data)
            _block.strategy=self.breed_strategy
            self.blocks.append(_block)
        
        self.activation_fun=pkl.load(data)

class RecurrentLayer(Layer):
    def __init__(self,input_size:int,output_size:int,block_number:int,block_nueron_number:int=64,block_population_size:int=512,init:Initializer=UniformInit(),breed_strategy=BreedStrategy()) -> None:
        super(RecurrentLayer,self).__init__(input_size+output_size,output_size,block_number,block_nueron_number,block_population_size,init,breed_strategy)

        self.last_outputs=np.zeros(output_size,dtype=np.float32)

    def step(self,_inputs:np.ndarray)->np.ndarray:
        outputs:np.ndarray=np.zeros(self.output_size,dtype=np.float32)
        
        inputs=np.concatenate((_inputs,self.last_outputs))

        for block in self.blocks:
            outputs+=block.fire(inputs)#/len(self.blocks)

        for n,activ in enumerate(self.activation_fun):
            outputs[n]=clip(activ(outputs[n]))

        self.last_outputs=np.copy(outputs)

        return outputs    

    def fire(self,_inputs:np.ndarray)->np.ndarray:
        outputs:np.ndarray=np.zeros(self.output_size,dtype=np.float32)
        
        inputs=np.concatenate((_inputs,self.last_outputs))

        for block in self.blocks:
            block.pickBatch()
            outputs+=block.fire(inputs)#/len(self.blocks)

        for n,activ in enumerate(self.activation_fun):
            outputs[n]=clip(activ(outputs[n]))

        self.last_outputs=np.copy(outputs)

        return outputs
    
    def save(self,memory:io.BufferedIOBase):
        '''
            Save each blocks
            Activation functions list
            Input size
            Output size
            Number of blocks
            Last output

            Every block will be saved in individual file
        '''
        # layer type 0x00 is recurrent type

        np.save(memory,np.array([0x00],dtype=np.int16))

        metadata=np.array([self.input_size,self.output_size,len(self.blocks)],dtype=np.int32)

        np.save(memory,metadata)

        np.save(memory,self.last_outputs)

        for block in self.blocks:
            block.save(memory)

        pkl.dump(self.activation_fun,memory)
 
    def load(self,data:io.RawIOBase):
        
        metadata=np.load(data)

        self.input_size=metadata[0]
        self.output_size=metadata[1]

        self.last_outputs=np.load(data)

        self.blocks.clear()

        for i in range(metadata[2]):
            _block=block.Block(0,0,0,0)
            _block.load(data)
            _block.strategy=self.breed_strategy
            self.blocks.append(_block)
        
        self.activation_fun=pkl.load(data)


LAYERS_TYPES_ID={
    0x00:RecurrentLayer,
    0x01:Layer
}