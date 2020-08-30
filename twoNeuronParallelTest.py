"""
Created on 09-Jul-2020

@author: Felix

expansion on 2-input single-neuron test case
has 3 inputs and 2 neurons in the same layer 
"""

from Neuron import Neuron
import numpy as np

t1 = 107
t2 = 31
t3 = 11

L1 = 2

# learning constant
q = 0.05


def updateMatrix(inputVector, outputVector, weights, inputLayerSize, outputLayerSize, learningConstant):
    return learningConstant * (np.transpose(inputVector) @ outputVector - ((weights @ np.transpose(outputVector) * np.identity(inputLayerSize)) @ np.tile(outputVector, (inputLayerSize, 1))))


layer = []
for i in range(L1):
    layer.append(Neuron())

w1 = np.random.random_sample((3, L1))

i1 = np.zeros((1, 3))
u1 = np.zeros((1, L1))
o1 = np.zeros((1, L1))

lim = 10000
x = 1

print(i1)
print(o1)
print(w1)

while lim >= x:

    # reset input to zero
    i1.fill(0)

    # check to see if inputs should fire
    if x % t1 == 0:
        i1[0][0] = 1
    if x % t2 == 0:
        i1[0][1] = 1
    if x % t3 == 0:
        i1[0][2] = 1

    u1 = i1 @ w1

    for i in range(L1):
        layer[i].spikeIn(u1[0][i])
        o1[0][i] = layer[i].updateNeuron()

    w1 += updateMatrix(i1, o1, w1, 3, L1, q)

    # increment iterator
    x += 1

print(w1)
