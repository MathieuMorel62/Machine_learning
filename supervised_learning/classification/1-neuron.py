#!/usr/bin/env python3
""" Class Neuron """

import numpy as np


class Neuron:
    """ Class Neuron performing binary classification """

    def __init__(self, nx):
        """ Settings for class Neuron """

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise TypeError("nx must be a positive integer")
        
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        return self.__W
    
    @property
    def b(self):
        return self.__b
    
    @property
    def A(self):
        return self.__A
