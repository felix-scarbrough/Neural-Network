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
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, importWeights):

        self.inputLayerSize, self.hiddenLayerSize, self.outputLayerSize = inputLayerSize, hiddenLayerSize, outputLayerSize

        # neuron hidden layers
        hiddenLayer, outputLayer = [], []
        for i in range(hiddenLayerSize):
            hiddenLayer.append(Neuron())

        for i in range(outputLayerSize):
            outputLayer.append(Neuron())

        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer

        if importWeights:
            print("Input filename to load weights from: ")
            filename = input()
            self.loadWeights(filename)
        else:
            self.w1 = np.random.random_sample((inputLayerSize, hiddenLayerSize))
            self.w2 = np.random.random_sample((hiddenLayerSize, outputLayerSize))

    # iterates the input countdowns and creates an input layer with values set to 1 if that input "neuron" is firing
    def inputUpdate(self, inputLayerSize, impulseCountDown, firingRates):
        print(inputLayerSize, impulseCountDown, firingRates)
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
        np.savez(filename, w1=self.w1, w2=self.w2)

    # loads the weights from a .npz file to the object
    def loadWeights(self, filename):
        data = np.load(filename)
        self.w1 = data['w1']
        self.w2 = data['w2']

    # main learning loop, runs with a single set of inputs for a fixed number of cycles, then saves the resulting
    # weights and layers
    def learningLoop(self, hiddenLayer, outputLayer, inputLayerSize, hiddenLayerSize, outputLayerSize, w1, w2, firingRates, learningConstant, limit):

        # create impulse countdown vector
        impulseCountDown = np.random.randint(10, size=firingRates.shape)

        # iterator
        x = 0

        # main loop
        while x <= limit:
            # test inputs to see if they should fire, then reset to maximum value, or decay current value.
            inputVector, impulseCountDown = self.inputUpdate(inputLayerSize, impulseCountDown, firingRates)

            # update the neuron layers
            outputVectorOne, hiddenLayerPotentials = self.updateLayer(hiddenLayer, inputVector, w1)
            outputVectorTwo, outputLayerPotentials = self.updateLayer(outputLayer, outputVectorOne, w2)

            # update weight matrices using Hebbian learning
            w1 = w1 + self.updateMatrix(inputVector, outputVectorOne, w1, inputLayerSize, hiddenLayerSize, learningConstant)
            w2 = w2 + self.updateMatrix(outputVectorOne, outputVectorTwo, w2, hiddenLayerSize, outputLayerSize, learningConstant)

            # print infomation at this step
            print(x, " : ", inputVector, outputVectorOne, outputVectorTwo)

            # increment
            x += 1

        # sets the object weights and layers to those produced by the learning loop, maintaining potential and weight
        # continuity between differant learning cycles
        self.w1, self.w2, self.hiddenLayer, self.outputLayer = w1, w2, hiddenLayer, outputLayer
