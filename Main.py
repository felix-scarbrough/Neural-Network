"""
Created on Mon Aug 31 2020

@author: Felix Scarbrough
"""

from NeuralNetwork import NeuralNetwork
import numpy as np

firingRateOne = np.array([[0.7, 0.1, 0.3, 0.2]])
firingRateTwo = np.array([[0.2, 0.3, 0.15, 0.8]])
firingRateThree = np.array([[0.2, 0.6, 0.3, 0.1]])
firingRateFour = np.array([[0.3, 0.5, 0.9, 0.3]])

learningConstant = 0.01
lim = 60

network = NeuralNetwork(4, 6, 6, 4, False)

for i in range(200):
    print(i, " : ")
    network.learningLoop(network.inputLayer, network.hiddenLayerOne, network.hiddenLayerTwo, network.outputLayer, network.inputLayerSize, network.hiddenLayerOneSize, network.hiddenLayerTwoSize, network.outputLayerSize, network.w1, network.w2, network.w3, firingRateOne, learningConstant, lim)
    network.learningLoop(network.inputLayer, network.hiddenLayerOne, network.hiddenLayerTwo, network.outputLayer, network.inputLayerSize, network.hiddenLayerOneSize, network.hiddenLayerTwoSize, network.outputLayerSize, network.w1, network.w2, network.w3, firingRateTwo, learningConstant, lim)
    network.learningLoop(network.inputLayer, network.hiddenLayerOne, network.hiddenLayerTwo, network.outputLayer, network.inputLayerSize, network.hiddenLayerOneSize, network.hiddenLayerTwoSize, network.outputLayerSize, network.w1, network.w2, network.w3, firingRateThree, learningConstant, lim)
    network.learningLoop(network.inputLayer, network.hiddenLayerOne, network.hiddenLayerTwo, network.outputLayer, network.inputLayerSize, network.hiddenLayerOneSize, network.hiddenLayerTwoSize, network.outputLayerSize, network.w1, network.w2, network.w3, firingRateFour, learningConstant, lim)
