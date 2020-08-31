# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Felix Scarbrough
"""

from Neuron import Neuron
import numpy as np

class NeuralNetwork:
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

    # function or applying spikes to neurons
    def applySpikes(self, layer, updateVector):
        for i in range(len(layer)):
            layer[i].spikeIn(updateVector[0][i])

    # function for updating neurons
    def updateNeurons(self, layer):
        neuronOutput = []
        for i in range(len(layer)):
            neuronOutput.append(layer[i].updateNeuron())
        return np.atleast_2d(neuronOutput)

    # function for returning neuron potentials
    def neuronPotentials(self, layer):
        potentials = []
        for neuron in layer:
            potentials.append(neuron.getPotential())
        return potentials

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
            # test inputs to see if they should fire, then reset to maximum value, or decay current value.
            for i in range(inputLayerSize):
                if impulseCountDown[0][i] == 0:
                    inputVector[0][i] = 1
                    impulseCountDown[0][i] = firingRates[0][i]
                else:
                    impulseCountDown[0][i] -= 1

            # create update vector
            updateVectorOne = inputVector @ w1

            # apply spikes to first layer of neurons and decay
            self.applySpikes(self, hiddenLayer, updateVectorOne)
            outputVectorOne = np.atleast_2d(self.updateNeurons(self, hiddenLayer))
            p1 = self.neuronPotentials(self, hiddenLayer)

            # create update vector
            updateVectorTwo = outputVectorOne @ w2

            # apply spikes to second layer of neurons and decay
            self.applySpikes(self, outputLayer, updateVectorTwo)
            outputVectorTwo = np.atleast_2d(self.updateNeurons(self, outputLayer))
            outputLayerPotentials = self.neuronPotentials(self, outputLayer)

            # update weight matrices using Hebbian learning
            w1 = w1 + self.updateMatrix(self, inputVector, outputVectorOne, w1, inputLayerSize, hiddenLayerSize, learningConstant)
            w2 = w2 + self.updateMatrix(self, outputVectorOne, outputVectorTwo, w2, hiddenLayerSize, outputLayerSize, learningConstant)

            # reset value of inputs to zero
            inputVector = np.zeros((1, inputLayerSize))

            # increment
            x += 1

