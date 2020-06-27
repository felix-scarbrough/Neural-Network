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
    lim = 10
    
    #iterator 
    x = 0
    
    #learning constant 
    Q = 0.35
    
    #input and hidden layer sizes 
    n0, n1, n2 = 4, 4, 4
    
    #weight matrices 
    w1 = Matrix(np.random.random_sample((n0, n1)))
    w2 = Matrix(np.random.random_sample((n1, n2)))
    
    #update matrices 
    um1, um2 = np.zeros((n0, n1)), np.zeros((n0, n1))
    
    #input, update, output, and potential vectorrs 
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
    def updateMatrix(q, a, b):
        inputVector = Matrix(a)
        outputVector = Transpose(Matrix(b))
        return q * inputVector * outputVector 
    
    #main loop
    while(x <= lim):
        #create random set of inputs
        for i in range(n0):
            i1[i] = np.random.randint(0, 2)
            
        #create update vector    
        u1 = np.matmul(i1, w1)
         
        #apply spikes to first layer of neurons and decay
        for i in range(n1):
            L1[i].spikeIn(u1[i])
            o1[i] = L1[i].updateNeuron()
            p1[i] = L1[i].getPotential()
        
        #create update vector
        u2 = np.matmul(o1, w2)
        
        #apply spikes to second layer of neurons and decay
        for i in range(n2):
            L2[i].spikeIn(u2[i])
            o2[i] = L2[i].updateNeuron()
            p2[i] = L2[i].getPotential()
        
        #update weight matrices using method 
        w1 = w1 + updateMatrix(Q, i1, o1)
        w2 = w2 + updateMatrix(Q, o1, o2)
        
        #print useful information at this step 
        print(x, ': ', i1, p1, o1, p1, o2)
        
        #incriment 
        x += 1 
          


        