import numpy as np
import neuron
import block
import layer
import network
from activation.sigmoid import Sigmoid
from activation.linear import Linear

import io

neuron1=neuron.Neuron(4,1)

neuron1.trails=24

print(neuron1.trails)
print(neuron1.input_weights)
print(neuron1.output_weights)

data=neuron1.dump()

neuron2=neuron.Neuron(4,1)

neuron2.load(data)

print(neuron2.trails)
print(neuron2.input_weights)
print(neuron2.output_weights)

# neuron saving/loading works

print("Block test: ")

block1=block.Block(4,1,16,512)

block1.createPopulation()

print(block1.population[0])
print(block1.population[-1])


mem=io.BytesIO()

block1.save(mem)

block2=block.Block(4,1,16,512)

print(mem.getbuffer().nbytes)

block2.load(io.BytesIO(mem.getvalue()))

print("Outputs")
print(block2.population[0])
print(block2.population[-1])

# block saving/loading works

print("Layer test")

layer1=layer.Layer(4,1,8,32,512)
layer1.setActivationFun([Sigmoid])

data=io.BytesIO()

layer1.save(data)

layer2=layer.Layer(0,0,0,0,0)

layer2.load(io.BytesIO(data.getvalue()))

print(layer2.input_size)
print(layer1.blocks[0].population[0])
print(layer2.blocks[0].population[0])

print(layer1.activation_fun)
print(layer2.activation_fun)

# layer saving seems to work,yeah

print("Network load/save test: ")

network1=network.Network(4)

network1.addLayer(4,4)

network1.addLayer(32,4)

network1.addLayer(2,2,[Sigmoid,Linear])

network.NetworkParser.save(network1,"network.bin")

print(network1.step([0.5,0.25,0.46,0.76]))

print(network1.layers[0].blocks[0].population[0])

network2=network.NetworkParser.load("network.bin")

print("Loaded network:")
print(network2.step([0.5,0.25,0.46,0.76]))
print(network2.layers[0].blocks[0].population[0])

# network saving/loading seems to work
