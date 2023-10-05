import numpy as np
import random
import timeit

x=[]
y=[]

for i in range(256):
    x.append(random.random())
    y.append(random.random())


dot=0

start=timeit.default_timer()


for i in range(256):
    dot+=x[i]*y[i]

print("Time: ",timeit.default_timer()-start," s")

x1=np.array(x)
y1=np.array(y)

start=timeit.default_timer()

dot=np.dot(x1,y1)

print("Time: ",timeit.default_timer()-start," s")