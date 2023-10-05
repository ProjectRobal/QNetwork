# Kapibara Control Network

## Introduction:

 A neuroevolution algorithm to control my social robot called Kapibara. 
It is based or inspired by neuroevolution algorithm such as:
- NEAT - a network wich can adjust it's size
- SANE - a network wich evolve neurons rather than entire network
- ESP - a SANE but with recurrent neurons
- CCMA - a algorithm wich utilize couple of cooperating networks 
- ECNN - a network wich not only return action to take but also estimation of quality of action

## Assumptions:

1. Algorithm uses population of neurons as genes. Every neuron has input, output weights also it has recurrent connection
wich means it uses it's past outputs like inputs. Since it is a neuroevolution algorithm it isn't vulnerable for
gradient explosion or vanishing gradient so it is using plain recurrent connection, neuron can store it's past output
and then use it to feed it to inputs. 

1. Neurons are also alowed to connect to neurons from upper layers affecting thiers internal recurrent memory.

1. Instead of crossing / mutate entiere networks we are going to apply evolution on neurons population.

1. There is also second type of population utilize wich is network schematics ( in ESP is called blueprints )
wich define topology of network. It holds reference to neurons it is composed of. Topology of network is
also going to be evalueted and evolved so we could find optimal size.

1. Network schematics will be grouped into species based on thier structure. They will compete with networks 
from the same species. Species are going to be also evalueted. It helps to decied wich network architecture is 
more well-performing.

1. Networks are going to have multiple hidden layers, made from batch of neurons ( genes ) with variable size.
Theta and tau, where theta is an amount of hidden layers and tau defines a size of neurons batch. 
Theta and tau is going to change according to evaluation trend ( how well the network is doing ).

1. Every networks are going to return action and Q - quality of that action. Network is trying to predict how good is
picked action. The Q value will be used in network evaluation with feedback from environment.

1. After action taken network will be evaluted based on feedback from environment and Q value. The evaluation will be applied 
evenly on every neuron network is made of.



## Proposition:

1. In a population of networks, a network is going to get information about evalutaion from network before.

1. Like NEAT changes it's structure adding neurons dynamically, my algorithm could use varing number of conncection
for each neuron.


## File description:

- neuron.py - store class that defines neuron object
- network.py - store class related to network and network template
- crossover.py - holds base class for neurons and networks crossover
- mutation.py - holds base class for neuron and network mutation
- dotproduct.py - holds base class for dot product calculation used by neurons
- specie.py - store class wich purpose is to hold objects with similar traits
- composer.py - store class wich tie everything together performing the algorithm
