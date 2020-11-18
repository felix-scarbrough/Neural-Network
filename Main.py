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

firingRateZero = np.array([[0, 0, 0, 0, 0, 0]])
firingRateOne = np.array([[0.6, 0.1, 0.03, 0.02, 0.1, 0.4]])
firingRateTwo = np.array([[0.02, 0.03, 0.015, 0.5, 0.7, 0.0]])
firingRateThree = np.array([[0.02, 0.7, 0.03, 0.1, 0.02, 0.4]])
firingRateFour = np.array([[0.03, 0.5, 0.7, 0.03, 0.5, 0.02]])

inputLayerSize = 6

'''firingRateOne = firingRate(inputLayerSize)
firingRateTwo = firingRate(inputLayerSize)
firingRateThree = firingRate(inputLayerSize)
firingRateFour = firingRate(inputLayerSize)'''

learningConstant = 0.001
lim = 30

network = NeuralNetwork(inputLayerSize, 8, 8, 4, learningConstant, False)

for i in range(100):
    network.learningLoop(firingRateOne, lim, 'red')
    network.learningLoop(firingRateTwo, lim, 'blue')
    network.learningLoop(firingRateThree, lim, 'yellow')
    network.learningLoop(firingRateFour, lim, 'green')

network.learningLoop(firingRateZero, 100, 'white')
network.learningLoop(np.array([[0.6, 0.0, 0.05, 0.01, 0.15, 0.5]]), lim, 'purple')
network.learningLoop(np.array([[0.05, 0.75, 0.02, 0.15, 0.03, 0.5]]), lim, 'orange')
network.learningLoop(np.array([[0.06, 0.65, 0.81, 0.01, 0.6, 0.03]]), lim, 'brown')
network.learningLoop(np.array([[0.04, 0.01, 0.005, 0.9, 0.5, 0.08]]), lim, 'pink')

x_pos, y_pos, colour = network.getData()
plt.scatter(x_pos, y_pos, c=colour)
plt.show()
