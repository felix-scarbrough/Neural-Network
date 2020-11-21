# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 2020
@author: Felix Scarbrough
"""

import math

# leaky-integrate-and-fire Neuron model that undergoes polarisation post-spike and has inhibited potential
class AdvancedNeuron:
    threshold = 1
    potential = 0
    spiked = False
    spikeTimer = 0
    decay = 0.97

    # returns the potential of the neuron - used for debugging
    def getPotential(self):
        return self.potential

    # applies a spike to the neuron
    def spikeIn(self, weight):
        if not self.spiked:
            self.potential += weight

    # check if the neuron has recently spiked.
    def checkSpiked(self):
        if self.spiked and self.spikeTimer >= 0:
            self.spikeTimer -= 1
        elif self.spiked:
            self.spiked = False
            self.spikeTimer = 0

    # checks if the neuron should spike, otherwise decays the potential
    def spikeOrDecay(self):
        if self.potential >= self.threshold:
            self.potential -= 1.2
            self.spiked = True
            self.spikeTimer = 5
            return 1
        else:
            self.potential = self.decay * self.potential
            return 0

    # returns 1 if the potential exceeds the threshold and sets potential to zero, otherwise decays the potential
    def updateNeuron(self):
        self.checkSpiked()
        return self.spikeOrDecay()






