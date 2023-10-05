import numpy as np

import neuron
from activation.sigmoid import Sigmoid

inputs=np.random.random(16)

neuron1=neuron.Neuron(16,2)

print(neuron1.fire(inputs))