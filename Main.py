"""
Created on Mon Aug 31 2020

@author: Felix Scarbrough
"""

from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import math

# calculate the average output of each neuron for a single input state using the number of spikes from that neuron during that state
def stateAverage(state, k):
    count = [0, 0, 0, 0]
    avg = [0, 0, 0, 0]
    for i in range(k):
        for j in range(4):
            count[j] += state[i][j]

    for i in range(4):
        avg[i] = count[i]/k
    return avg, count

# calculate the standard deviation for each individual neuron during a single input state from it's output and average
def standardDiv(state, average, k):
    stanDiv = [0, 0, 0, 0]
    for i in range(k):
        for j in range(4):
            stanDiv[j] = (state[i][j] - average[j])**2
    for i in range(4):
        stanDiv[i] = math.sqrt(stanDiv[i]/k)
    return stanDiv

# determine the total standard deviation across all neurons for a single state
def totalStanDiv(stanDiv, count):
    stanDivTotal = 0
    for i in range(4):
        if count[i] != 0:
            stanDivTotal += (stanDiv[i] ** 2) / count[i]
    return math.sqrt(stanDivTotal)

# set the number of learning iterations (n), testing iterations (k), and the learning loop limit (lim)
n = 100
k = 10
lim = 50

# set input layer size and learning constant
inputLayerSize = 6
learningConstant = 0.01


# set firing rates
firingRateZero = np.array([[0, 0, 0, 0, 0, 0]])
firingRateOne = np.array([[0.9, 0.1, 0.03, 0.02, 0.1, 0.6]])
firingRateTwo = np.array([[0.02, 0.03, 0.015, 0.7, 0.8, 0.0]])
firingRateThree = np.array([[0.02, 0.8, 0.03, 0.1, 0.02, 0.4]])
firingRateFour = np.array([[0.03, 0.5, 0.9, 0.03, 0.6, 0.02]])

# create lists to store total spikes for each neuron during each input state
inputOne = []
inputTwo = []
inputThree = []
inputFour = []

# create neural network object
network = NeuralNetwork(inputLayerSize, 8, 8, 4, learningConstant, False, "networkOutputTwo.txt")

# learning loop
for i in range(n):
    network.learningLoop(firingRateOne, lim, 'red', False)
    network.learningLoop(firingRateTwo, lim, 'blue', False)
    network.learningLoop(firingRateThree, lim, 'yellow', False)
    network.learningLoop(firingRateFour, lim, 'green', False)

# loop with no input to clear neuron potentials
network.learningLoop(firingRateZero, 100, 'white', False)

# testing loop
for i in range(k):
    inputOne.append(network.learningLoop(np.array([[0.9, 0.0, 0.05, 0.01, 0.15, 0.5]]), lim, 'purple', True).flatten().tolist())
    inputTwo.append(network.learningLoop(np.array([[0.05, 0.75, 0.02, 0.15, 0.03, 0.5]]), lim, 'orange', True).flatten().tolist())
    inputThree.append(network.learningLoop(np.array([[0.06, 0.65, 0.81, 0.01, 0.6, 0.03]]), lim, 'brown', True).flatten().tolist())
    inputFour.append(network.learningLoop(np.array([[0.04, 0.01, 0.005, 0.9, 0.5, 0.08]]), lim, 'pink', True).flatten().tolist())

# calculate input state averages and total counts
inputOneAvg, inputOneCount = stateAverage(inputOne, k)
inputTwoAvg, inputTwoCount = stateAverage(inputTwo, k)
inputThreeAvg, inputThreeCount = stateAverage(inputThree, k)
inputFourAvg, inputFourCount = stateAverage(inputFour, k)

# calculate input state standard deviations for each neuron
inputOneStanDiv = standardDiv(inputOne, inputOneAvg, k)
inputTwoStanDiv = standardDiv(inputTwo, inputTwoAvg, k)
inputThreeStanDiv = standardDiv(inputThree, inputThreeAvg, k)
inputFourStanDiv = standardDiv(inputFour, inputFourAvg, k)

# calculate total standard deviation for each input state across all neurons
inputOneTotalStanDiv = totalStanDiv(inputOneStanDiv, inputOneCount)
inputTwoTotalStanDiv = totalStanDiv(inputTwoStanDiv, inputTwoCount)
inputThreeTotalStanDiv = totalStanDiv(inputThreeStanDiv, inputThreeCount)
inputFourTotalStanDiv = totalStanDiv(inputFourStanDiv, inputFourCount)

# store the counts, averages, neuron standard deviations, and total standard deviations to file
f = open("networkResults.txt", "w")

f.write("Results for Basic Neuron" + "\n")
f.write("\n")

f.write("input state spike counts:" + "\n")
f.write(str(inputOneCount) + "\n")
f.write(str(inputTwoCount) + "\n")
f.write(str(inputThreeCount) + "\n")
f.write(str(inputFourCount) + "\n")
f.write("\n")

f.write("input state spike averages:" + "\n")
f.write(str(inputOneAvg) + "\n")
f.write(str(inputTwoAvg) + "\n")
f.write(str(inputThreeAvg) + "\n")
f.write(str(inputFourAvg) + "\n")
f.write("\n")

f.write("input state neuron standard deviations:" + "\n")
f.write(str(inputOneStanDiv) + "\n")
f.write(str(inputTwoStanDiv) + "\n")
f.write(str(inputThreeStanDiv) + "\n")
f.write(str(inputFourStanDiv) + "\n")
f.write("\n")

f.write("input state total standard deviation:" + "\n")
f.write(str(inputOneTotalStanDiv) + "\n")
f.write(str(inputTwoTotalStanDiv) + "\n")
f.write(str(inputThreeTotalStanDiv) + "\n")
f.write(str(inputFourTotalStanDiv) + "\n")

f.close()

# create scatter plot of neuron spikes
x_pos, y_pos, colour = network.getData()
plt.scatter(x_pos, y_pos, c=colour)
plt.xlabel('Step')
plt.title('Neural Network firing output with 1000 training iterations, 10 testing iterations, learning loop limit of 50 and a learning constant of 0.01')
plt.show()
