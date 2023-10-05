'''

    A class that holds methods responsible for crossover and mutation.
    A object of class is hold by network class and then it is passed to layers and then 
    to blocks.

'''
import neuron

from base.crossover import Crossover
from base.mutation import Mutation

#from crossover.onepoint import OnePoint
from crossover.qonepoint import QOnePoint
from mutation.gaussmutation import GaussMutaion


class BreedStrategy:
    def __init__(self,cross:Crossover=QOnePoint,mutate:Mutation=GaussMutaion):
        self.cross=cross
        self.mutate=mutate

    def crossover(self,neuron1:neuron.Neuron,neuron2:neuron.Neuron)->neuron.Neuron:
        return self.cross(neuron1,neuron2)
    
    def mutation(self,neuron:neuron.Neuron)->neuron.Neuron:
        return self.mutate(neuron)
    