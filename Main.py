"""
Created on Mon Aug 31 2020

@author: Felix Scarbrough
"""

from NeuralNetwork import NeuralNetwork
import numpy as np

firingRateOne = np.array([[3, 5, 15, 15]])
firingRateTwo = np.array([[5, 3, 9, 15]])
firingRateThree = np.array([[9, 5, 3, 9]])
firingRateFour = np.array([[15, 9, 5, 3]])
network = NeuralNetwork(4, 4, 4, True)

for i in range(5):
    network.learningLoop(network.hiddenLayer, network.outputLayer, network.inputLayerSize, network.hiddenLayerSize, network.outputLayerSize, network.w1, network.w2, firingRateOne, 0.05, 50)
    network.learningLoop(network.hiddenLayer, network.outputLayer, network.inputLayerSize, network.hiddenLayerSize, network.outputLayerSize, network.w1, network.w2, firingRateTwo, 0.05, 50)
    network.learningLoop(network.hiddenLayer, network.outputLayer, network.inputLayerSize, network.hiddenLayerSize, network.outputLayerSize, network.w1, network.w2, firingRateThree, 0.05, 50)
    network.learningLoop(network.hiddenLayer, network.outputLayer, network.inputLayerSize, network.hiddenLayerSize, network.outputLayerSize, network.w1, network.w2, firingRateFour, 0.05, 50)
network.saveWeights("networkweightsTest.npz")
