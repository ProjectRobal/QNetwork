import numpy as np
import neuron
import block
from timeit import default_timer
import time

block1=block.Block(2,1,64,512)

inputs=np.random.random(2)

block1.createPopulation()

block1.pickBatch()

i=0

def eval(inputs:np.ndarray,outputs:np.ndarray):
    return np.exp(-abs((np.linalg.norm(inputs)-np.linalg.norm(outputs)))*0.01)*100

inputs=np.array([10,10])

for k in range(1000000):
    #start=default_timer()

    #inputs=np.random.random(2)

    block1.pickBatch()

    outputs=block1.fire(inputs)

    q=eval(inputs,outputs)
    
    print(q)

    block1.Evaluate(q)

    block1.Mating()

    #print("dT: ",default_timer()-start," s")
    time.sleep(0.1)    

print("After ",k," trials")
