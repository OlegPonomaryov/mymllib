"""Implementations of different mathematical functions."""
import numpy as np


def sigmoid(z):
    """Return sigmoid function value for product of X and theta."""

    # A value of exp(n) exceeds a capacity of double-precision floating-point variables if n is higher than
    # approximately 709.7. For np.exp() this results in warning message and inf return value, which also makes the
    # sigmoid function to return 0. This may cause errors, because, for instance, cost function of logistic regression
    # calculates logarithm of sigmoid's return value, which cannot be calculated for 0. To avoid this, all values of z
    # that are lower than -709.7 (because z is used with '-' in np.exp()) are replaced with -709.7.
    z = np.maximum(z, -709.7)

    h = 1 / (1 + np.exp(-z))

    # Values that are very close to 1 (like 0.9999999999999999999999) cannot be stored in double-precision floating-
    # point variables due to their significant digits limitation and are rounded to 1. This may cause errors too,
    # because logistic regression cost function also calculates logarithm of 1 - sigmoid(z), so all return values that
    # are equal to 1 are replaced with the largest representable floating point value that is less than 1.
    return np.minimum(h, 0.9999999999999999)


def log_loss(predicted, actual):
    """Return log loss value."""
    return -np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)) / predicted.shape[0]
