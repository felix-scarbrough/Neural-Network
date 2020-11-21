"""
Created on 09-Jul-2020

@author: Felix

expansion on 2-input single-neuron test case
has 3 inputs and 2 neurons in the same layer 
"""

from Neuron import Neuron
import numpy as np

# code made to test behaviour of two neurons in single layer receiving inputs

t1 = 107
t2 = 31
t3 = 11

L1 = 2

# learning constant
q = 0.05

# learning algorithm
def updateMatrix(inputVector, outputVector, weights, inputLayerSize, outputLayerSize, learningConstant):
    return learningConstant * (np.transpose(inputVector) @ outputVector - ((weights @ np.transpose(outputVector) * np.identity(inputLayerSize)) @ np.tile(outputVector, (inputLayerSize, 1))))

# create neuron layer
layer = []
for i in range(L1):
    layer.append(Neuron())

# create initial weights
w1 = np.random.random_sample((3, L1))

# create input, update, and output vectors
i1 = np.zeros((1, 3))
u1 = np.zeros((1, L1))
o1 = np.zeros((1, L1))

# learning limit and global iterator
lim = 10000
x = 1

# print initial values of input and output vectors, and weight matrix
print(i1)
print(o1)
print(w1)

# learning loop
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

    # create update vector from input vector and weight matrix
    u1 = i1 @ w1

    # apply spikes to neurons using update vector
    for i in range(L1):
        layer[i].spikeIn(u1[0][i])
        o1[0][i] = layer[i].updateNeuron()

    # update the weight matrix using the learning algorithm
    w1 += updateMatrix(i1, o1, w1, 3, L1, q)

    # increment iterator
    x += 1

# print the final weight matrix
print(w1)
