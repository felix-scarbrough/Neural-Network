# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020
@author: Felix Scarbrough
"""

import math

class Neuron:
    threshold = 1
    potential = 0
    groundState = 0
    spiked = False
    spikeTimer = 0
    decayConst = 0.05

    # returns the potential of the neuron - used for debugging
    def getPotential(self):
        return self.potential

    # applies a spike to the neuron
    def spikeIn(self, weight):
        self.potential += weight

    def checkSpiked(self):
        if self.spiked and self.spikeTimer <= 0:
            self.spiked = False
            self.spikeTimer = 0
            self.decayConst = 0.2
            self.threshold = 1
        elif self.spiked:
            self.spikeTimer -= 1

    def spikeOrDecay(self):
        if self.potential >= self.threshold:
            self.spiked = True
            self.spikeTimer = 5
            self.potential -= 1.2
            self.threshold = 100
            self.decayConst = 0.4
            return 1
        else:
            self.potential = math.exp(-self.decayConst) * (self.potential - self.groundState) + self.groundState
            return 0

    # returns 1 if the potential exceeds the threshold and sets potential to zero, otherwise decays the potential
    def updateNeuron(self):
        self.checkSpiked()
        return self.spikeOrDecay()






