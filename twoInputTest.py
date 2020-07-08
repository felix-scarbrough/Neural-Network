'''
Created on 07-Jul-2020

@author: Felix
'''
from Neuron import Neuron
import numpy as np 

neuron = Neuron()

weights = np.random.random_sample((2, 1))
inputVector = np.zeros((1, 2))
updateVector = np.zeros(1)
updateValue = 0
neuronSpike = 0
q = 0.05

lim = 10000 
x = 1
t1 = 12
t2 =  5

weightSum = 0

print(weights)

while(lim >= x):
    
    for i in range(1):
        inputVector[i] = 0
        
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

    
