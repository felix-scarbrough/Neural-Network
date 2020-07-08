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
    Q = 0.05
    
    #input and hidden layer sizes 
    n0, n1, n2 = 5, 3, 6
    
    #weight arrays 
    w1 = np.random.random_sample((n0, n1))
    w2 = np.random.random_sample((n1, n2))
    
    #update arrays 
    um1, um2 = np.zeros((n0, n1)), np.zeros((n0, n1))
    
    #input, update, output, and potential vectors 
    i1 = np.zeros((1, n0))
    u1,  p1, o1 = np.zeros((1, n1)), np.zeros((1, n1)), np.zeros((1, n1))
    u2, p2, o2 = np.zeros((1, n2)), np.zeros((1, n2)), np.zeros((1, n2))
    
    
    #neuron hidden layers 
    L1, L2 = [], []
    for i in range(n1):
        L1.append(Neuron())

    for i in range(n2):
        L2.append(Neuron())
        
    #returns the matrix product of the input and output vectors scaled to the learning constant (Q)
    def updateMatrix(learningConstant, inputVector, outputVector):
        return learningConstant * (inputVector * np.transpose(outputVector))
         
    
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
            i1[0][i] = np.random.randint(0, 2)
            
        #create update vector    
        u1 = i1 @ w1  
         
        #apply spikes to first layer of neurons and decay 
        applySpikes(L1, u1)
        o1 = updateNeurons(L1)
        p1 = neuronPotentials(L1)
        
        #create update vector
        u2 = o1 @ w2
        
        #apply spikes to second layer of neurons and decay
        applySpikes(L2, u2)
        o2 = updateNeurons(L2)
        p2 = neuronPotentials(L2)
        
        #update weight matrices using method 
        w1 = w1 + updateMatrix(Q, i1, o1)
        w2 = w2 + updateMatrix(Q, o1, o2)
        
        #renormalize weight matrices
        weightSumOne = w1.sum(axis=1)
        w1 = w1/weightSumOne
        
        weightSumTwo = w2.sum(axis=1)
        w2 = w2/weightSumTwo
        
        #print useful information at this step 
        print(x, ': ', i1, p1, o1, p2, o2)
        print(w1)
        print(w2)
        print()
        
        #incriment 
        x += 1 
          


        