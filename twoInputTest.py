'''
Created on 07-Jul-2020

@author: Felix
'''
from Neuron import BasicNeuron as Neuron
import numpy as np 

# testing environment to check neuron operation, currently configured for examining learning algorithm.
neuron = Neuron()

# creates weights and vectors
weights = np.random.random_sample((2, 1))
inputVector = np.zeros((1, 2))
updateVector = np.zeros(1)
updateValue = 0
neuronSpike = 0
q = 0.05

# iteration limit, global count, and firing periods
lim = 10000 
x = 1
t1 = 12
t2 =  5

weightSum = 0

print(weights)

# main operation loop
while(lim >= x):
    
    inputVector.fill(0)
        
    if((x % t1) == 0):
        inputVector[0][0] = 1
        
    if((x % t2) == 0):
        inputVector[0][1] = 1
        
    neuron.spikeIn(inputVector @ weights)
    neuronSpike = neuron.updateNeuron()
    
    weightSum = (inputVector @ weights).item()
    weights += q * weightSum * (np.transpose(inputVector) - (weightSum * weights))  
    x += 1 

print(weights)
