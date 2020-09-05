# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 2020
@author: Felix Scarbrough
"""


class Neuron:
    threshold = 1
    potential = 0

    # returns the potential of the neuron - used for debugging
    def getPotential(self):
        return self.potential

    # applies a spike to the neuron
    def spikeIn(self, weight):
        self.potential += weight

    # returns 1 if the potential exceeds the threshold and sets potential to zero, otherwise decays the potential
    def updateNeuron(self):
        if self.potential >= self.threshold:
            self.potential = 0
            return 1
        else:
            self.potential = 0.95 * self.potential
            return 0
