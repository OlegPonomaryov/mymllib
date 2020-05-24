"""Implementations of different mathematical functions."""
import numpy as np


def sigmoid(z):
    """Return sigmoid function value for product of X and theta."""
    h = 1 / (1 + safe_exp(-z))

    # Values that are very close to 1 (like 0.9999999999999999999999) cannot be stored in double-precision floating-
    # point variables due to their significant digits limitation and are rounded to 1. This may cause errors too,
    # because logistic regression cost function also calculates logarithm of 1 - sigmoid(z), so all return values that
    # are equal to 1 are replaced with the largest representable floating point value that is less than 1.
    return np.minimum(h, 0.9999999999999999)


def log_loss(predicted, actual):
    """Return logistic (sigmoid) loss function value."""
    return -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))


def log_cost(predicted, actual):
    """Return logistic (sigmoid) cost function value (without regularization)."""
    return np.sum(log_loss(predicted, actual)) / predicted.shape[0]


def softmax(z):
    """Return softmax function value for product of X and theta."""
    t = safe_exp(z)
    h = t / t.sum(axis=1, keepdims=True)
    return h


def softmax_loss(predicted, actual):
    """Return softmax loss function value."""
    # Because softmax function may return exactly 0, a small value added to its result in order to avoid attempts to
    # calculate log(0)
    return -(actual * np.log(predicted + 1E-15))


def softmax_cost(predicted, actual):
    """Return softmax cost function value (without regularization)."""
    return np.sum(softmax_loss(predicted, actual)) / predicted.shape[0]


def safe_exp(z):
    # A value of exp(n) exceeds a capacity of double-precision floating-point variables if n is higher than
    # approximately 709.7. For np.exp() this results in warning message and inf return value, which may cause further
    # problems when using it (like making sigmoid function return 0 which results in an attempt to calculate log(0) in
    # its cost function). To avoid this, all values of z that are higher than 709.7 are replaced with 709.7.
    z = np.minimum(z, 709.7)
    return np.exp(z)

