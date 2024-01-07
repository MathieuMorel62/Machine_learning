#!/usr/bin/env python3
""" Function that converts a one-hot matrix into a vector of labels """
import numpy as np


def one_hot_decode(one_hot):
  """ converts a one-hot matrix into a vector of labels """
  if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
    return None

  labels = np.argmax(one_hot, axis=0)

  return labels
