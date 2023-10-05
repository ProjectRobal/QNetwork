'''
Notes:

Instead of using equation Q(t)=Q(t-1)+alfa*R(t) use 
Q(t)=Q(t-1)+alfa*d(R(t)/Q(t-1))*R(t)

Where d is function that is near 0 when (Q(t) / Q(t-1)) is less than 1.0.

'''

import os
import numpy as np
import gymnasium

from network import Network,NetworkParser
from layer import RecurrentLayer,Layer
from BreedStrategy import BreedStrategy

from activation.sigmoid import Sigmoid
from activation.linear import Linear
from activation.relu import Relu
from initializer.gaussinit import GaussInit

from buffer import TrendBuffer

import matplotlib.pyplot as pyplot

env=gymnasium.make("CartPole-v1",render_mode="human")

init=GaussInit(0,0.1)

network1=Network(2)

network1.addLayer(2,4,RecurrentLayer,[Relu,Relu],init,(8,64))

evaluation_trend:TrendBuffer=TrendBuffer(20)

epsilon=0.1

trends:float=[]

best_eval=0

def trendfunction(eval:float,network:Network)->float:

    trend:float=evaluation_trend.trendline()
    global epsilon
    global best_eval

    #print("Trend: ",trend)

    evaluation_trend.push(eval)

    #_epsilon=np.exp(2.3*(eval/500))*0.1

    _epsilon=eval/200

    if eval>best_eval:
        best_eval=eval
        print("Best current evaluation: ",best_eval)

    if _epsilon>epsilon:
        print("New epsilon:",_epsilon)
        epsilon=_epsilon
 
    return epsilon


def state_to_reward(observation:np.ndarray):
    p=observation[0]
    theta=observation[2]
    return  - (p**2)/11.52 - (theta**2)/288

if os.path.exists("tests/checkpoint/last.pk"):
    print("Loading checkpoint!!")
    #It doesn't load for some reason
    #network1=NetworkParser.load("tests/checkpoint/last.pk")

network1.setTrendFunction(trendfunction)


(state,_) = env.reset()

EPISODE_NUMBER=1000

rewards=[]

action=0

inputs:np.ndarray=np.array([state[0],state[2]])

episode=0

best_score=0

while True:
    
    env.render()

    terminated=False

    steps=0

    steps_list:list[float]=[]

    network1.shuttle()

    for i in range(1):

        steps=0

        while not terminated:
            output=network1.run([state[0],state[2]])
            action:int=np.argmax(output)

            state,reward,terminated,truncated,info=env.step(action)

            steps+=1

        terminated=False
        (state,_)=env.reset()
        
        steps_list.append(steps)

    evaluation=np.mean(steps_list)

    steps_list.clear()

    #print("Episode: ",episode, "finished with average ",evaluation," steps")

    rewards.append(evaluation)

    network1.evalute(evaluation)
    #print("Reward: ",evaluation)
    #print("Epsilon: ",network1.layers[0].blocks[0].epsilon)

    #if steps > best_score:
    #    best_score=steps
    NetworkParser.save(network1,"tests/checkpoint/last.pk")

    episode+=1

env.close()

pyplot.plot(np.arange(0,EPISODE_NUMBER,1),rewards)

pyplot.show()