#!/usr/bin/env python3
""" Class Neuron """

import numpy as np


class Neuron:
    """ Class Neuron performing binary classification"""

    def __init__(self, nx):
        """ Settings for class Neuron """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise TypeError("nx must be a positive integer")
        
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
