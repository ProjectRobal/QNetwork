from typing import Callable
import io
import numpy as np
import pickle as pkl

from base.activation import Activation
from base.initializer import Initializer

from activation.linear import Linear

from BreedStrategy import BreedStrategy
from initializer.uniforminit import UniformInit

import layer

TrendFunction=Callable[[float,object],float]

class Network:
    
    '''
        A class that defines network.
        It stores layers wich defines hidden layers of network.
    '''
    def __init__(self,input_size:int,breed_strategy:BreedStrategy=BreedStrategy()):
        self.input_size=input_size
        self.breed_strategy=breed_strategy

        self.layers:list[layer.Layer]=[]

        # copy of best performed layer
        self.best_layer:list[layer.Layer]=[]

        self.trend_function:TrendFunction=None

    def setTrendFunction(self,trend_function:TrendFunction):
        self.trend_function=trend_function

    def resetPopulation(self):
        for layer in self.layers:
            pass
    
    def getLayerBestRatioPopulation(self,id:int)->float:

        if id>=len(self.layers) or id<0:
            id=0

        return self.layers[id].getBestRatioPopulation()


    def addLayer(self,output_size:int,block_number:int,layer_type=layer.Layer,activation:list[Activation]=[],init:Initializer=UniformInit(),block_params:tuple[int,int]=(16,512)):
        '''
            High level function to add layer to network
        '''
        if len(activation)==0:
            activation=[Linear]*output_size

        if len(self.layers)==0:
            input_size=self.input_size
        else:
            input_size=self.layers[-1].output_size

        _layer=layer_type(input_size,output_size,block_number,block_params[0],block_params[1],init,self.breed_strategy)

        _layer.setActivationFun(activation)

        self.add_layers(_layer)

    def add_layers(self,layers:list[layer.Layer]|layer.Layer):
        '''
            Low level function to add layer to network
        '''
        if isinstance(layers,layer.Layer):
            self.layers.append(layers)
            return

        for l in layers:
            self.layers.append(l)

    def shuttle(self):
        '''
        Pick a new batch in layers
        '''
        for l in self.layers:
            l.pickNBatch()

    def run(self,inputs:np.ndarray)->np.ndarray:
        '''
        Run without picking new batch
        '''
        
        for l in self.layers:
            output=l.step(inputs)

            inputs=output

        for l in self.layers:

            l.mate()

        return output

    def step(self,inputs:np.ndarray)->np.ndarray:
        '''
        Run with picking new batch
        '''
        
        for l in self.layers:
            output=l.fire(inputs)

            inputs=output

        for l in self.layers:

            l.mate()

        return output
    
    def evalute(self,eval:float):

        if self.trend_function is not None:
            dpopulation=self.trend_function(eval,self)
            for l in self.layers:
                l.changeBestRatioPopulation(dpopulation)
        
        eval=eval/len(self.layers)

        for l in self.layers:
            l.evalute(eval)


class NetworkParser:
    '''
        A helper class used for saving/loading network
    '''

    @staticmethod
    def save(network:Network,filename:str):
        with open(filename,"wb+") as file:
            metadata=np.array([network.input_size,len(network.layers)])
            np.save(file,metadata)

            pkl.dump(network.breed_strategy,file)

            for _layer in network.layers:
                layer_m=io.BytesIO()

                _layer.save(layer_m)

                file.write(layer_m.getvalue())
            
            file.close()
    
    @staticmethod
    def load(filename:str)->Network:
        network=Network(0)

        with open(filename,"rb") as file:
            metadata=np.load(file)
            network.input_size=metadata[0]
            network.breed_strategy=pkl.load(file)

            for i in range(metadata[1]):
                type:int=np.load(file)[0]

                layer_t=layer.LAYERS_TYPES_ID[type]

                _layer=layer_t(0,0,0,0,0)
                _layer.breed_strategy=network.breed_strategy

                _layer.load(file)

                network.add_layers(_layer)

            file.close()

        return network



