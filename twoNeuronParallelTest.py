'''
Created on 09-Jul-2020

@author: Felix

expansion on 2-input single-neuron test case
has 3 inputs and 2 neurons in the same layer 
'''

from Neuron import Neuron
import numpy as np 

t1 = 5
t2 = 9
t3 = 7

layerSize = 4

#learning constant 
q = 0.05

layer = [] 
for i in range(layerSize):
    layer.append(Neuron())
    
weights = np.random.random_sample((3, layerSize))
updateMatrix = np.zeros((3, layerSize))
    
inputVector = np.zeros((1, 3))
updateVector = np.zeros((1, layerSize))
outputVector = np.zeros((1, layerSize))

lim = 5000
x = 1 

print(weights)
    
while (lim >= x):
    
    #reset input to zero 
    inputVector.fill(0)
    
    #check to see if inputs should fire
    if ((x % t1) == 0):
        inputVector[0][0] = 1
    if ((x % t2) == 0):
        inputVector[0][1] = 1
    if ((x % t3) == 0):
        inputVector[0][2] = 1
    
    updateVector = inputVector @ weights    
    
    for i in range(layerSize): 
        layer[i].spikeIn(updateVector[0][i])
        outputVector[0][i] = layer[i].updateNeuron()
    
    weights += q * np.transpose(updateVector * np.identity(layerSize) @ np.transpose(np.transpose(np.tile(inputVector, (layerSize, 1))) - np.transpose((updateVector * np.identity(layerSize) @ np.transpose(weights)))))
    
    #incriment iterator 
    x += 1 

print(weights)

