# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Felix Scarbrough
"""

from Neuron import Neuron
import numpy as np


class Main:
    # number of iterations for the network
    lim = 30

    # iterator
    x = 0

    # learning constant
    Q = 0.05

    # input and hidden layer sizes
    n0, n1, n2 = 4, 8, 4

    # firing rate array
    firingRates = np.array([[3, 5, 7, 11]])
    impulseCountDown = np.random.randint(10, size=firingRates.shape)

    # weight arrays
    w1 = np.random.random_sample((n0, n1))
    w2 = np.random.random_sample((n1, n2))

    # update arrays
    um1, um2 = np.zeros((n0, n1)), np.zeros((n0, n1))

    # input, update, output, and potential vectors
    i1 = np.zeros((1, n0))
    u1, p1, o1 = np.zeros((1, n1)), np.zeros((1, n1)), np.zeros((1, n1))
    u2, p2, o2 = np.zeros((1, n2)), np.zeros((1, n2)), np.zeros((1, n2))

    # neuron hidden layers
    L1, L2 = [], []
    for i in range(n1):
        L1.append(Neuron())

    for i in range(n2):
        L2.append(Neuron())

    # returns the matrix product of the input and output vectors scaled to the learning constant (Q)
    def updateMatrix(inputVector, outputVector, weights, inputLayerSize, outputLayerSize, learningConstant):
        return learningConstant * (np.transpose(inputVector) @ outputVector - ((weights @ np.transpose(outputVector) * np.identity(inputLayerSize)) @ np.tile(outputVector, (inputLayerSize, 1))))

    def updateFiringRates(self, firingRates):
        self.firingRates = firingRates

    def saveWeights(self, filename):
        np.savez(filename, w1=self.w1, w2=self.w2)

    def loadWeights(self, filename):
        data = np.load(filename)
        self.w1 = data['w1']
        self.w2 = data['w2']


    # function and vectorized function for applying spikes to neurons
    applySpike = lambda a, b: a.spikeIn(b)
    applySpikes = np.vectorize(applySpike)

    # function and vectorized function for updating neurons
    updateNeuron = lambda a: a.updateNeuron()
    updateNeurons = np.vectorize(updateNeuron)

    # function and vectorized function for returning neuron potentials
    neuronPotential = lambda a: a.getPotential()
    neuronPotentials = np.vectorize(neuronPotential)

    # main loop
    while x <= lim:

        # test inputs to see if they should fire, then reset to maximum value, or decay current value.
        for i in range(n0):
            if impulseCountDown[0][i] == 0:
                i1[0][i] = 1
                impulseCountDown[0][i] = firingRates[0][i]
            else:
                impulseCountDown[0][i] -= 1

        # create update vector
        u1 = i1 @ w1

        # apply spikes to first layer of neurons and decay
        applySpikes(L1, u1)
        o1 = np.atleast_2d(updateNeurons(L1))
        p1 = neuronPotentials(L1)

        # create update vector
        u2 = o1 @ w2

        # apply spikes to second layer of neurons and decay
        applySpikes(L2, u2)
        o2 = np.atleast_2d(updateNeurons(L2))
        p2 = neuronPotentials(L2)

        # update weight matrices using Hebbian learning
        w1 = w1 + updateMatrix(i1, o1, w1, n0, n1, Q)
        w2 = w2 + updateMatrix(o1, o2, w2, n1, n2, Q)

        # reset value of inputs to zero
        i1 = np.zeros((1, n0))

        # increment
        x += 1
