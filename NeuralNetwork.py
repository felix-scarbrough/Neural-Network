# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Felix Scarbrough
"""

from Neuron import Neuron
import numpy as np


class NeuralNetwork:
    # placeholder filename
    filename = "testfile.npz"

    # number of iterations for the network
    lim = 30

    # learning constant
    Q = 0.05

    # input and hidden layer sizes
    n0, n1, n2 = 4, 8, 4

    # firing rate array
    firingRates = np.array([[3, 5, 7, 11]])

    # returns the matrix product of the input and output vectors scaled to the learning constant (Q)
    def updateMatrix(self, inputVector, outputVector, weights, inputLayerSize, outputLayerSize, learningConstant):
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
    def learningLoop(self, inputLayerSize, hiddenLayerSize, outputLayerSize, firingRates, learningConstant, limit, importWeights):

        # create update matrices
        um1, um2 = np.zeros((inputLayerSize, hiddenLayerSize)), np.zeros((hiddenLayerSize, outputLayerSize))

        # input, update, output, and potential vectors
        inputVector = np.zeros((1, inputLayerSize))
        updateVectorOne, hiddenLayerPotentials, outputVectorOne = np.zeros((1, hiddenLayerSize)), np.zeros((1, hiddenLayerSize)), np.zeros((1, hiddenLayerSize))
        updateVectorTwo, outputLayerPotentials, outputVectorTwo = np.zeros((1, outputLayerSize)), np.zeros((1, outputLayerSize)), np.zeros((1, outputLayerSize))

        # neuron hidden layers
        hiddenLayer, outputLayer = [], []
        for i in range(hiddenLayerSize):
            hiddenLayer.append(Neuron())

        for i in range(outputLayerSize):
            outputLayer.append(Neuron())

        # create impulse countdown vector
        impulseCountDown = np.random.randint(10, size=firingRates.shape)

        if importWeights:
            print("Input filename to load weights from: ")
            filename = input()
            self.loadWeights(self, filename)
        else:
            w1 = np.random.random_sample((inputLayerSize, hiddenLayerSize))
            w2 = np.random.random_sample((hiddenLayerSize, outputLayerSize))

        # iterator
        x = 0

        # print initial weights
        # print(w1, w2)

        while x <= limit:
            print(x)
            # test inputs to see if they should fire, then reset to maximum value, or decay current value.
            for i in range(inputLayerSize):
                if impulseCountDown[0][i] == 0:
                    inputVector[0][i] = 1
                    impulseCountDown[0][i] = firingRates[0][i]
                else:
                    impulseCountDown[0][i] -= 1

            # create update vector
            u1 = inputVector @ w1

            # apply spikes to first layer of neurons and decay
            self.applySpikes(hiddenLayer, u1)
            outputVectorOne = np.atleast_2d(self.updateNeurons(hiddenLayer))
            p1 = self.neuronPotentials(hiddenLayer)

            # create update vector
            updateVectorTwo = outputVectorOne @ w2

            # apply spikes to second layer of neurons and decay
            self.applySpikes(outputLayer, updateVectorTwo)
            outputVectorTwo = np.atleast_2d(self.updateNeurons(outputLayer))
            outputLayerPotentials = self.neuronPotentials(outputLayer)

            # update weight matrices using Hebbian learning
            w1 = w1 + self.updateMatrix(self, inputVector, outputVectorOne, w1, inputLayerSize, hiddenLayerSize, learningConstant)
            w2 = w2 + self.updateMatrix(self, outputVectorOne, outputVectorTwo, w2, hiddenLayerSize, outputLayerSize, learningConstant)

            # debug infomation
            print(x, " : ", inputVector, outputVectorOne, outputVectorTwo)

            # reset value of inputs to zero
            inputVector = np.zeros((1, inputLayerSize))

            # increment
            x += 1

        # print final weights
        # print(w1, w2)
