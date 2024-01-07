#!/usr/bin/env python3
""" Function that converts a numeric label vector into a one-hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
  """ Converts a numeric label vector into a one-hot matrix """
  if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
    return None

  m = Y.shape[0]
  one_hot_matrix = np.zeros((classes, m))

  # Filling the one-hot matrix
  for idx, label in enumerate(Y):
    if label >= classes:
      return None
    one_hot_matrix[label, idx] = 1

  return one_hot_matrix
