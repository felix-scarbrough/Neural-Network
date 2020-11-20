# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020
@author: Felix Scarbrough
"""

import math

# Neuron that when spiked has it's potential polarised before decaying to the ground state and has a short inhibitory period
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

    def checkSpiked(self):
        if self.spiked and self.spikeTimer >= 0:
            self.spikeTimer -= 1
        elif self.spiked:
            self.spiked = False
            self.spikeTimer = 0

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






