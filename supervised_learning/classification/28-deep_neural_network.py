#!/usr/bin/env python3
""" Module to create a deep neural network """
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
  """ Deep neural network performing binary classification """
  def __init__(self, nx, layers, activation='sig'):
    if type(nx) != int:
      raise TypeError("nx must be an integer")
    if nx < 1:
      raise ValueError("nx must be a positive integer")

    if type(layers) != list or not layers:
      raise TypeError("layers must be a list of positive integers")
    if any(type(i) != int or i <= 0 for i in layers):
      raise TypeError("layers must be a list of positive integers")

    if activation not in ['sig', 'tanh']:
      raise ValueError("activation must be 'sig' or 'tanh'")
        

    self.__L = len(layers)
    self.__cache = {}
    self.__weights = {}
    self.__activation = activation

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

  @property
  def activation(self):
    return self.__activation
  
  def sigmoid(self, Z):
    """ Sigmoid activation function. """
    return 1 / (1 + np.exp(-Z))

  def tanh(self, Z):
    """ Tanh activation function. """
    return np.tanh(Z)

  def softmax(self, Z):
    """ Softmax activation function. """
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

  def forward_prop(self, X):
    """ Forward propagation for the neural network. """
    self.__cache['A0'] = X

    for l in range(1, self.__L + 1):
      W = self.__weights['W' + str(l)]
      b = self.__weights['b' + str(l)]
      A_prev = self.__cache['A' + str(l - 1)]

      Z = np.dot(W, A_prev) + b
      if l == self.__L:
        A = self.softmax(Z)
      else:
        if self.__activation == 'sig':
          A = self.sigmoid(Z)
        elif self.__activation == 'tanh':
          A = self.tanh(Z)

      self.__cache['A' + str(l)] = A

    return A, self.__cache


  def cost(self, Y, A):
    """ Calculates the cost using cross-entropy. """
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(A))
    return cost

  def evaluate(self, X, Y):
    """ Evaluates the neural network's predictions. """
    A, _ = self.forward_prop(X)
    cost = self.cost(Y, A)
    return A, cost

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
        A_prev = cache['A' + str(l - 1)]
        if self.__activation == 'sig':
          dZ = dA_prev * (A_prev * (1 - A_prev))
        elif self.__activation == 'tanh':
          dZ = dA_prev * (1 - A_prev ** 2)

  def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
    """ Trains the deep neural network. """
    if not isinstance(iterations, int):
      raise TypeError("iterations must be an integer")
    if iterations <= 0:
      raise ValueError("iterations must be a positive integer")
    if not isinstance(alpha, float):
      raise TypeError("alpha must be a float")
    if alpha <= 0:
      raise ValueError("alpha must be positive")
    if (verbose or graph) and (not isinstance(step, int) or step <= 0 or step > iterations):
      raise TypeError("step must be an integer and positive and <= iterations")

    costs = []
    for i in range(iterations + 1):
      A, _ = self.forward_prop(X)
      self.gradient_descent(Y, self.__cache, alpha)

      if i % step == 0 or i == iterations:
        cost = self.cost(Y, A)
        costs.append(cost)
        if verbose:
          print(f"Cost after {i} iterations: {cost}")

    if graph:
      plt.plot(range(0, iterations + 1, step), costs)
      plt.xlabel("iteration")
      plt.ylabel("cost")
      plt.title("Training Cost")
      plt.show()

    return A, cost
  
  def save(self, filename):
    """ Saves the instance object to a file in pickle format. """
    if not filename.endswith('.pkl'):
      filename += '.pkl'

    with open(filename, 'wb') as file:
      pickle.dump(self, file)

  @staticmethod
  def load(filename):
    """ Loads a pickled DeepNeuralNetwork object. """
    try:
      with open(filename, 'rb') as file:
        return pickle.load(file)
    except FileNotFoundError:
      return None
