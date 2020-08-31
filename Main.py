from NeuralNetwork import NeuralNetwork
import numpy as np

firingRates = np.array([[3, 5, 7, 11]])
network = NeuralNetwork

network.learningLoop(network, 4, 8, 4, firingRates, 0.05, 30, False)
