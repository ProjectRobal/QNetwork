'''
    A file that store class wich purpose is to keep object like neurons and networks with similar traits.
'''

from network import Network

'''
 You can also use function to calculate similarity factor, then store (factor,network) in dictionary,
sort it and slpit them.

'''

class Specie:
    '''
        A class that store objects with similar traits.
        In case of neurons it can be correlation between weights arrays.

        In case of networks it is tau,theta,correlation of neurons weights etc.
    '''
    def __init__(self):
        self.objects:list[Network]=[]

    def doMatch(self,network:Network)->bool:
        theta=network.theta
        tau=network.tau
        

    def Append(self,network:Network):
        self.objects.append(network)

    def addIfMatch(self,network:Network)->bool:
        '''
            Add network to specie if matches criteria,
            return true if network has been added to specie
            or false otherwise.
        '''

        if network.specie_ptr is None and self.doMatch(network):
            network.specie_ptr=self
            self.objects.append(network)
            return True
        
        return False