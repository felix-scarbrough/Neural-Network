# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020

@author: Felix Scarbrough
"""


from Neuron import Neuron
from sympy import *
import numpy as np

class Main():

    #number of iterations for the network 
    lim = 30
    
    #iterator 
    x = 0
    
    #learning constant 
    Q = 0.35
    
    #input and hidden layer sizes 
    n0, n1, n2 = 4, 4, 4
    
    #weight arrays 
    w1 = np.matrix(np.random.random_sample((n0, n1)))
    w2 = np.matrix(np.random.random_sample((n1, n2)))
    
    #update arrays 
    um1, um2 = np.zeros((n0, n1)), np.zeros((n0, n1))
    
    #input, update, output, and potential vectors 
    i1 = np.zeros(n0)
    u1,  p1, o1 = np.zeros(n1), np.zeros(n1), np.zeros(n1)
    u2, p2, o2 = np.zeros(n2), np.zeros(n2), np.zeros(n2)
    
    
    #neuron hidden layers 
    L1, L2 = [], []
    for i in range(n1):
        L1.append(Neuron())

    for i in range(n2):
        L2.append(Neuron())
        
    #returns the matrix product of the input and output vectors scaled to the learning constant (Q)
    def updateWeights(q, a, b):
        N = a.size
        inputVector = Matrix(a)
        outputVector = Transpose(Matrix(b))
        updateMatrix = np.array(q * inputVector * outputVector).astype(np.float64)
        weightSum = updateMatrix.sum()
        return updateMatrix - np.full(updateMatrix.shape, weightSum/N)
        
         
    
    #function and vectorized function for applying spikes to neurons 
    applySpike = lambda a, b: a.spikeIn(b)
    applySpikes = np.vectorize(applySpike)
    
    #function and vectorized function for updating neurons  
    updateNeuron = lambda a: a.updateNeuron()
    updateNeurons = np.vectorize(updateNeuron)
    
    #function and vectorized function for returning neuron potentials 
    neuronPotential = lambda a: a.getPotential()
    neuronPotentials = np.vectorize(neuronPotential)
    
    #main loop
    while(x <= lim):
        #create random set of inputs
        for i in range(n0):
            i1[i] = np.random.randint(0, 2)
            
        #create update vector    
        u1 = np.matmul(i1, w1)
         
        #apply spikes to first layer of neurons and decay 
        applySpikes(L1, u1)
        o1 = updateNeurons(L1)
        p1 = neuronPotentials(L1)
        
        #create update vector
        u2 = np.matmul(o1, w2)
        
        #apply spikes to second layer of neurons and decay
        applySpikes(L2, u2)
        o2 = updateNeurons(L2)
        p2 = neuronPotentials(L2)
        
        #update weight matrices using method 
        w1 += updateWeights(Q, i1, o1)
        w2 += updateWeights(Q, o1, o2)
        
        #print useful information at this step 
        print(x, ': ', w1, w2)
        print()
        
        #incriment 
        x += 1 
          


        