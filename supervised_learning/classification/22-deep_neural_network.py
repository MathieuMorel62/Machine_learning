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

    self.__L = len(layers)
    self.__cache = {}
    self.__weights = {}

    # Initialization of weights and biases
    for l in range(1, self.__L + 1):
      self.__weights['b' + str(l)] = np.zeros((layers[l - 1], 1))

      layer_size = layers[l - 1]
      if l == 1:
        he_val = np.sqrt(2 / nx)
        self.__weights['W' + str(l)] = np.random.randn(layer_size, nx) * he_val
      else:
        he_val = np.sqrt(2 / layers[l - 2])
        self.__weights['W' + str(l)] = np.random.randn(layer_size, layers[l - 2]) * he_val

  @property
  def L(self):
    return self.__L

  @property
  def cache(self):
    return self.__cache

  @property
  def weights(self):
    return self.__weights

  def sigmoid(self, Z):
    """ Sigmoid activation function. """
    return 1 / (1 + np.exp(-Z))

  def forward_prop(self, X):
    """ Performs the front propagation of the neural network. """
    self.__cache['A0'] = X

    for l in range(1, self.__L + 1):
      W = self.__weights['W' + str(l)]
      b = self.__weights['b' + str(l)]
      A_prev = self.__cache['A' + str(l - 1)]

      Z = np.dot(W, A_prev) + b
      A = self.sigmoid(Z)

      self.__cache['A' + str(l)] = A

    return A, self.__cache

  def cost(self, Y, A):
    """ Calculates the cost of the model using logistic regression. """
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
    return cost

  def evaluate(self, X, Y):
    """ Evaluates the predictions of the neural network. """
    A, _ = self.forward_prop(X)
    cost = self.cost(Y, A)
    predictions = np.where(A >= 0.5, 1, 0)
    return predictions, cost

  def gradient_descent(self, Y, cache, alpha=0.05):
    """ Performs one pass of gradient descent on the neural network. """
    m = Y.shape[1]
    A = cache['A' + str(self.__L)]
    dZ = A - Y

    for l in reversed(range(1, self.__L + 1)):
      A_prev = cache['A' + str(l - 1)]
      W = self.__weights['W' + str(l)]
      b = self.__weights['b' + str(l)]

      dW = (1 / m) * np.dot(dZ, A_prev.T)
      db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
      dA_prev = np.dot(W.T, dZ)

      # Update weights and biases
      self.__weights['W' + str(l)] -= alpha * dW
      self.__weights['b' + str(l)] -= alpha * db

      if l > 1:
        dZ = dA_prev * (A_prev * (1 - A_prev))

  def train(self, X, Y, iterations=5000, alpha=0.05):
    """ Trains the deep neural network. """
    if not isinstance(iterations, int):
      raise TypeError("iterations must be an integer")
    if iterations <= 0:
      raise ValueError("iterations must be a positive integer")
    if not isinstance(alpha, float):
      raise TypeError("alpha must be a float")
    if alpha <= 0:
      raise ValueError("alpha must be positive")

    for _ in range(iterations):
      A, cache = self.forward_prop(X)
      self.gradient_descent(Y, cache, alpha)

    return self.evaluate(X, Y)
