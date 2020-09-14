"""
Created on Mon Aug 31 2020

@author: Felix Scarbrough
"""

from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

def firingRate(size):
    firingRate = np.random.rand(1, size)
    return firingRate

firingRateOne = np.array([[0.9, 0.1, 0.03, 0.02, 0.1, 0.4]])
firingRateTwo = np.array([[0.02, 0.03, 0.015, 0.8, 0.7, 0.0]])
firingRateThree = np.array([[0.02, 0.9, 0.03, 0.1, 0.02, 0.4]])
firingRateFour = np.array([[0.03, 0.5, 0.9, 0.03, 0.5, 0.2]])

inputLayerSize = 6

'''firingRateOne = firingRate(inputLayerSize)
firingRateTwo = firingRate(inputLayerSize)
firingRateThree = firingRate(inputLayerSize)
firingRateFour = firingRate(inputLayerSize)'''

learningConstant = 0.005
lim = 100

network = NeuralNetwork(inputLayerSize, 8, 8, 4, learningConstant, False)

for i in range(1000):
    network.learningLoop(firingRateOne, lim, 'red')
    network.learningLoop(firingRateTwo, lim, 'blue')
    network.learningLoop(firingRateThree, lim, 'yellow')
    network.learningLoop(firingRateFour, lim, 'green')

x_pos, y_pos, colour = network.getData()
plt.scatter(x_pos, y_pos, c=colour)
plt.show()
