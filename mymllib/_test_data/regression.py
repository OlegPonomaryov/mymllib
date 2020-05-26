"""Test data for classification models."""
import numpy as np

_X = [[3, 5],
      [6.1, 4.7],
      [3, 1],
      [-2, 5.4],
      [12.3, 0],
      [-4, -2],
      [3, 7],
      [15, 0.8],
      [-5, 2],
      [42, 13],
      [8.005, -27],
      [11, 7]]
_y = (10 + np.sum(_X * np.asarray([1, 3]), axis=1)).tolist()  # y = 10 + x1 + 3*x2

X_train, y_train = _X[:8], _y[:8]
X_test, y_test = _X[8:], _y[8:]
