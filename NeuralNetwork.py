# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Felix Scarbrough
"""

from Neuron import Neuron
import numpy as np

class NeuralNetwork:

    # Set input layer size, hidden layer size, and output layer size on creation, set if to import weights, and
    # create neuron layers
    def __init__(self, inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize, importWeights):

        self.inputLayerSize, self.hiddenLayerOneSize, self.hiddenLayerTwoSize, self.outputLayerSize = inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize

        # neuron layers
        inputLayer, hiddenLayerOne, hiddenLayerTwo, outputLayer = self.createLayer(inputLayerSize), self.createLayer(hiddenLayerOneSize), self.createLayer(hiddenLayerTwoSize), self.createLayer(outputLayerSize)

        self.inputLayer = inputLayer
        self.hiddenLayerOne = hiddenLayerOne
        self.hiddenLayerTwo = hiddenLayerTwo
        self.outputLayer = outputLayer

        if importWeights:
            print("Input filename to load weights from: ")
            filename = input()
            self.loadWeights(filename)
        else:
            self.w1 = np.random.random_sample((inputLayerSize, hiddenLayerOneSize))
            self.w2 = np.random.random_sample((hiddenLayerOneSize, hiddenLayerTwoSize))
            self.w3 = np.random.random_sample((hiddenLayerTwoSize, outputLayerSize))


    def createLayer(self, layerSize):
        layer = []
        for i in range(layerSize):
            neuron = Neuron()
            layer.append(neuron)
        return layer

    # iterates the input countdowns and creates an input layer with values set to 1 if that input "neuron" is firing
    def inputUpdate(self, inputLayerSize, impulseCountDown, firingRates):
        inputVector = np.zeros(firingRates.shape)
        for i in range(inputLayerSize):
            if impulseCountDown[0][i] == 0:
                inputVector[0][i] = 1
                impulseCountDown[0][i] = firingRates[0][i]
            else:
                impulseCountDown[0][i] -= 1
        return inputVector, impulseCountDown

    # creates the update vector from the input vector and the weights for that layer, and applies it to the layer using
    # the applySpikes method. Updates the neuron layer and creates the output vector using the updateNeurons method, and
    # creates the layer Potentials array using the neuronPotentials method, then returns the output vector and layer
    # potentials
    def updateLayer(self, layer, inputVector, weights):
        # create update vector
        updateVector = inputVector @ weights

        # apply spikes to second layer of neurons and decay
        self.applySpikes(layer, updateVector)
        outputVector = self.updateNeurons(layer)
        layerPotentials = self.neuronPotentials(layer)

        return outputVector, layerPotentials

    # applies the input value from the update vector to each neuron in the layer as an input spike
    def applySpikes(self, layer, updateVector):
        for i in range(len(layer)):
            layer[i].spikeIn(updateVector[0][i])

    # checks each neuron for if it's potential exceeds the threshold, firing if it does. otherwise decay neurons, then
    # return the output vector
    def updateNeurons(self, layer):
        neuronOutput = []
        for i in range(len(layer)):
            neuronOutput.append(layer[i].updateNeuron())
        return np.atleast_2d(neuronOutput)

    # gets the current potential from each neuron in the layer then returns it as an array
    def neuronPotentials(self, layer):
        potentials = []
        for neuron in layer:
            potentials.append(neuron.getPotential())
        return potentials

    # returns the matrix product of the input and output vectors scaled to the learning constant (Q)
    def updateMatrix(self, inputVector, outputVector, weights, inputLayerSize, outputLayerSize, learningConstant):
        return learningConstant * (np.transpose(inputVector) @ outputVector - ((weights @ np.transpose(outputVector) * np.identity(inputLayerSize)) @ np.tile(outputVector, (inputLayerSize, 1))))

    # change the current value of the firingRates array (currently not-used)
    def updateFiringRates(self, firingRates):
        self.firingRates = firingRates

    # save the object weights to a .npz file
    def saveWeights(self, filename):
        np.savez(filename, w1=self.w1, w2=self.w2, w3=self.w3)

    # loads the weights from a .npz file to the object
    def loadWeights(self, filename):
        data = np.load(filename)
        self.w1 = data['w1']
        self.w2 = data['w2']
        self.w3 = data['w3']

    # main learning loop, runs with a single set of inputs for a fixed number of cycles, then saves the resulting
    # weights and layers
    def learningLoop(self, inputLayer, hiddenLayerOne, hiddenLayerTwo, outputLayer, inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize, w1, w2, w3, firingRates, learningConstant, limit):
        # identity matrix for input Layer
        w0 = np.identity(inputLayerSize)

        # test code
        runningTotal = np.zeros((1, outputLayerSize))

        # iterator
        x = 0

        # main loop
        while x <= limit:
            # test inputs to see if they should fire, then reset to maximum value, or decay current value.

            # update the neuron layers
            inputVector, inputLayerPotentials = self.updateLayer(inputLayer, firingRates, w0)
            outputVectorOne, hiddenLayerOnePotentials = self.updateLayer(hiddenLayerOne, inputVector, w1)
            outputVectorTwo, hiddenLayerTwoPotentials = self.updateLayer(hiddenLayerTwo, outputVectorOne, w2)
            outputVectorThree, outputLayerPotentials = self.updateLayer(outputLayer, outputVectorTwo, w3)

            # update weight matrices using Hebbian learning
            w1 = w1 + self.updateMatrix(inputVector, outputVectorOne, w1, inputLayerSize, hiddenLayerOneSize, learningConstant)
            w2 = w2 + self.updateMatrix(outputVectorOne, outputVectorTwo, w2, hiddenLayerOneSize, hiddenLayerTwoSize, learningConstant)
            w3 = w3 + self.updateMatrix(outputVectorTwo, outputVectorThree, w3, hiddenLayerTwoSize, outputLayerSize, learningConstant)

            # print information at this step
            # print(x, " : ", inputVector, outputVectorOne, outputVectorTwo, outputVectorThree)
            runningTotal += outputVectorThree

            # increment
            x += 1

        # sets the object weights and layers to those produced by the learning loop, maintaining potential and weight
        # continuity between different learning cycles
        # runningTotal = runningTotal / limit
        print(runningTotal)
        self.w1, self.w2, self.w3, self.inputLayer, self.hiddenLayerOne, self.hiddenLayerTwo, self.outputLayer = w1, w2, w3, inputLayer, hiddenLayerOne, hiddenLayerTwo, outputLayer
