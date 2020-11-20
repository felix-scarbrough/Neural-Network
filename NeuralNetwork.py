# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Felix Scarbrough
"""

from Neuron import BasicNeuron as Neuron
import numpy as np

class NeuralNetwork:

    # Set input layer size, hidden layer size, and output layer size on creation, set if to import weights, and
    # create neuron layers
    def __init__(self, inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize, learningConstant, importWeights, saveFile):

        # layer dimensions
        self.inputLayerSize, self.hiddenLayerOneSize, self.hiddenLayerTwoSize, self.outputLayerSize = inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize

        # create and store neuron layers
        inputLayer, hiddenLayerOne, hiddenLayerTwo, outputLayer = self.createLayer(inputLayerSize), self.createLayer(hiddenLayerOneSize), self.createLayer(hiddenLayerTwoSize), self.createLayer(outputLayerSize)
        self.inputLayer, self.hiddenLayerOne, self.hiddenLayerTwo, self.outputLayer = inputLayer, hiddenLayerOne, hiddenLayerTwo, outputLayer

        self.learningConstant = learningConstant
        self.saveFile = saveFile

        # imports weights from file if option is selected
        if importWeights:
            print("Input filename to load weights from: ")
            filename = input()
            self.loadWeights(filename)

        # otherwise create random starting weights
        else:
            self.w1 = np.random.random_sample((inputLayerSize, hiddenLayerOneSize))
            self.w2 = np.random.random_sample((hiddenLayerOneSize, hiddenLayerTwoSize))
            self.w3 = np.random.random_sample((hiddenLayerTwoSize, outputLayerSize))

        #create metacount global step tracker for plot and lists to store neuron spike times.
        self.metaCount = 0
        self.x_pos, self.y_pos, self.colour = [], [], []

    # method to create a layer of neurons
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

    # gets the spike time data from the neural network object for scatter plot
    def getData(self):
        return self.x_pos, self.y_pos, self.colour

    # main learning loop, runs with a single set of inputs for a fixed number of cycles, then saves the resulting weights and layers
    def learningLoop(self, firingRates, limit, colour, getTotal):

        # import the metacount from the object
        metaCount = self.metaCount

        #create local iterator
        x = 0

        # create local variables from layers and their sizes
        inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize = self.inputLayerSize, self.hiddenLayerOneSize, self.hiddenLayerTwoSize, self.outputLayerSize
        inputLayer, hiddenLayerOne, hiddenLayerTwo, outputLayer = self.inputLayer, self.hiddenLayerOne, self.hiddenLayerTwo, self.outputLayer

        # create local variable from the learning constant
        learningConstant = self.learningConstant

        # assign weights to local variables and create identity matrix for input layer
        w0 = np.identity(inputLayerSize)
        w1, w2, w3 = self.w1, self.w2, self.w3

        # test code
        runningTotal = np.zeros((1, outputLayerSize))

        # main loop
        while x <= limit:
            # update the neuron layers
            inputVector, inputLayerPotentials = self.updateLayer(inputLayer, firingRates, w0)
            outputVectorOne, hiddenLayerOnePotentials = self.updateLayer(hiddenLayerOne, inputVector, w1)
            outputVectorTwo, hiddenLayerTwoPotentials = self.updateLayer(hiddenLayerTwo, outputVectorOne, w2)
            outputVectorThree, outputLayerPotentials = self.updateLayer(outputLayer, outputVectorTwo, w3)

            # update weight matrices using Hebbian learning
            w1 = w1 + self.updateMatrix(inputVector, outputVectorOne, w1, inputLayerSize, hiddenLayerOneSize, learningConstant)
            w2 = w2 + self.updateMatrix(outputVectorOne, outputVectorTwo, w2, hiddenLayerOneSize, hiddenLayerTwoSize, learningConstant)
            w3 = w3 + self.updateMatrix(outputVectorTwo, outputVectorThree, w3, hiddenLayerTwoSize, outputLayerSize, learningConstant)

            # saves the spike time data to the scatter plot lists
            for i in range(outputLayerSize):
                if outputVectorThree[0][i] == 1:
                    runningTotal[0][i] += 1
                    self.y_pos.append(i)
                    self.x_pos.append(metaCount)
                    self.colour.append(colour)

            f = open(self.saveFile, "a")
            f.write(str(outputVectorThree.flatten().tolist()) + "\n")
            f.close()

            # increment
            x += 1
            metaCount += 1

        # continuity between different learning cycles
        # runningTotal = runningTotal / limit
        # print(runningTotal)
        self.w1, self.w2, self.w3, self.inputLayer, self.hiddenLayerOne, self.hiddenLayerTwo, self.outputLayer, self.metaCount = w1, w2, w3, inputLayer, hiddenLayerOne, hiddenLayerTwo, outputLayer, metaCount

        # returns the total of spikes for each neuron from the learning loop
        if getTotal:
            return runningTotal
