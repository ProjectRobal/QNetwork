import numpy as np
import layer
import math

import timeit

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

'''

Now we have problem with nan value which is solve yeah!

I need to think about a way to favorize the best network configuration.
Because we are getting bunch of networks that are meah and then we got one 
the best of the bounch.
I think to store N best networks and when we get apropiete amount of networks
perform crossover over thier populations.

When we increase block number and decrese neuron number in each, the network comes 
to best solution faster.

'''

# readings from sensors plus compressed audio spectogram, outputs: motor output power and three action (froward,backward,stop)
layer1=layer.Layer(6,256,32,16,512)

layer2=layer.Layer(256,2,16,16,512)

# linear regression problem

points=[(-1.0,1.0),(-0.73,1.54),(-0.43,2.14)]

inputs=np.array([-1.0,1.0,-0.73,1.54,-0.43,2.14],dtype=np.float32)

def regression_test(a:float,b:float)->float:
    error:float=0.0

    for x,y in points:
        _y=a*x+b
        error+=(y-_y)**2
    
    return error

def error_to_rewrd(e:float)->float:

    return np.exp(-np.abs(e)*0.01)*100.0

best_val=-1000

for n in range(1000000):
    #start=timeit.default_timer()

    output=layer2.fire(layer1.fire(inputs))

    error=regression_test(output[0],output[1])

    eval=error_to_rewrd(error)

    if eval>best_val:
        best_val=eval
        print("Best reward after: ",n," steps")
        print(eval)

    #print("Output reward",eval)

    layer1.evalute(eval/2.0)
    layer2.evalute(eval/2.0)

    blockPrint()

    layer2.mate()
    layer1.mate()

    enablePrint()

    #print("Time: ",timeit.default_timer()-start," s")
