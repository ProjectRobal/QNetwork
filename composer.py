'''
    A file with class that will tie everything together. It will manage evaluation, crossovers, muatation
    between neurons and networks. Divide populations into species.

    Basic working of algorithm loop:

    1. Initialize population of neurons with random weights
    2. Put all of them into one initial specie object
    3. Initialize population of networks with random parameters
    4. Pick first network from networks population 
    Loop:
        1. Attach neurons to  network by sampling a batch of neurons population
            Network_steps:
                1. Perform action, inputs-> network -> (action,Q)
                2. Evaluate action based on feedback from environemt and Q value
                3. If number of steps exced max_step go to Next_network else go to Network_steps
                Next_network:
                    1. Calculate network evaluation as the mean of step evaluation
                    2. Pass the evaluation evenly to neurons
                    3. Choose next network from networks population
                    4. If end of networks population hasn't been reach go to Loop
        2. Do evolution on neurons population among each species
        3. Do evolution on network population among each species
        4. Sometimes pick best neurons from each species and do evolution among them
        5. Sort neurons into species
        6. Sort networks into species
        7. Set network population at the begining and go to Loop

'''

'''

To do:
Dot product function
Crossover function
Mutation (Guasian mutation)

Layer class
Network class
Species class, sort networks base on structure

Neuron,Networks,Composer saving/loading
Composer loop and evaluation

'''

import numpy as np

from base.dotproduct import Product
from base.mutation import Mutation
from base.crossover import Crossover

from dotproducts.dotnumpy import NumpyDotProduct
from mutation.gaussmutation import GaussMutaion
from crossover.onepoint import OnePoint

from neuron import Neuron
from network import Network
from specie import Specie

def check_kwargs(name:str,default,**kwargs):
    if name in kwargs.keys():
        return kwargs[name]
    
    return default

class Composer:

    '''
        A class that is responsible for networks,neurons evaluation for crossover and muatation
    '''

    def __init__(self,network_input_size:int,network_output_size:int,**kwargs):
        self.network_input_size=network_input_size
        self.network_output_size=network_output_size

        self.neuron_population_size:int=check_kwargs("neuron_population_size",1024,kwargs)

        self.network_population_size:int=check_kwargs("network_population_size",20,kwargs)

        # a max size of a neuron batch
        self.max_tau:int=check_kwargs("max_tau",256,kwargs)

        # max amount of hidden layers
        self.max_theta:int=check_kwargs("max_theta",1,kwargs)

        # a initial size of a neuron batch, zero means that initial population will have random tau from (1,max_tau)
        self.init_tau:int=check_kwargs("tau",0,kwargs)

        # a initial amount of hidden layers, zero means that initial population will have random theta from (1,max_theta)
        self.init_theta:int=check_kwargs("theta",0,kwargs)

        # an amount of steps each network will perform in environemt
        self.step_time:int=check_kwargs("step_time",20,kwargs)

        self.dot_product:Product=NumpyDotProduct

        self.mutation:Mutation=GaussMutaion

        self.crossover:Crossover=OnePoint

        self.PrepareNeuronPopulation()

        self.PrepareNetworkPopulation()

    def setDotProductMethod(self,dot_prod:Product):
        self.dot_product=dot_prod

    def setMutationMethod(self,mutation:Mutation):
        self.mutation=mutation

    def setCrossoverMethod(self,crossover:Crossover):
        self.crossover=crossover

    def CreateNetwork(self)->Network:
        if self.init_theta==0:
            theta=int(np.random.random()*self.max_theta)
        else:
            theta=self.init_theta

        if self.init_tau==0:
            tau=int(np.random.random()*self.max_tau)
        else:
            tau=self.init_tau

        return Network(self.network_input_size,self.network_output_size,theta,tau)
    
    def PrepareNeuronPopulation(self):
        self.neurons:list[Neuron]=[Neuron(self.network_input_size,self.network_output_size,self.dot_product)]*self.neuron_population_size

    def PrepareNetworkPopulation(self):
        self.species:list[Specie]=[Specie()]

        self.species[0].Append(self.CreateNetwork())



