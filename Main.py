"""
Created on Mon Aug 31 2020

@author: Felix Scarbrough
"""

from NeuralNetwork import NeuralNetwork
import numpy as np

firingRateOne = np.array([[0.7, 0.1, 0.3, 0.2, 0.1, 0.5]])
firingRateTwo = np.array([[0.2, 0.3, 0.15, 0.8, 0.7, 0.0]])
firingRateThree = np.array([[0.2, 0.6, 0.3, 0.1, 0.2, 0.9]])
firingRateFour = np.array([[0.3, 0.5, 0.9, 0.3, 0.5, 0.2]])

learningConstant = 0.01
lim = 100

network = NeuralNetwork(6, 8, 8, 4, learningConstant, False)

for i in range(200):
    print(i, " : ")
    network.learningLoop(firingRateOne, lim)
    network.learningLoop(firingRateTwo, lim)
    network.learningLoop(firingRateThree, lim)
    network.learningLoop(firingRateFour, lim)
