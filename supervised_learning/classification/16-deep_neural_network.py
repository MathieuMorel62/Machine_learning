#!/usr/bin/env python3
""" Module to create a deep neural network """
import numpy as np


class DeepNeuralNetwork:
    """ Deep neural network performing binary classification """
    def __init__(self, nx, layers):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or not layers:
            raise TypeError("layers must be a list of positive integers")
        if any(type(i) != int or i <= 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialization of weights and biases
        for l in range(1, self.L + 1):
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))

            layer_size = layers[l - 1]
            if l == 1:
                he_val = np.sqrt(2 / nx)
                self.weights['W' + str(l)] = np.random.randn(layer_size, nx) * he_val
            else:
                he_val = np.sqrt(2 / layers[l - 2])
                self.weights['W' + str(l)] = np.random.randn(layer_size, layers[l - 2]) * he_val
