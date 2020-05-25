"""Implementations of different mathematical functions."""
import numpy as np


def sigmoid(x):
    h = 1 / (1 + safe_exp(-x))

    # Values that are very close to 1 (like 0.9999999999999999999999) cannot be stored in double-precision floating-
    # point variables due to their significant digits limitation and are rounded to 1. This may cause errors too,
    # because logistic regression cost function also calculates logarithm of 1 - sigmoid(x), so all return values that
    # are equal to 1 are replaced with the largest representable floating point value that is less than 1.
    return np.minimum(h, 0.9999999999999999)


def log_loss(predicted, actual):
    """Return logistic (sigmoid) loss function value."""
    return -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))


def log_cost(predicted, actual):
    """Return logistic (sigmoid) cost function value (without regularization)."""
    return np.sum(log_loss(predicted, actual)) / predicted.shape[0]


def softmax(x):
    t = safe_exp(x)
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


def tanh(x):
    exp_plus, exp_minus = safe_exp(x), safe_exp(-x)
    return (exp_plus - exp_minus) / (exp_plus + exp_minus)


def safe_exp(x):
    # A value of exp(n) exceeds a capacity of double-precision floating-point variables if n is higher than
    # approximately 709.7. For np.exp() this results in warning message and inf return value, which may cause further
    # problems when using it (like making sigmoid function return 0 which results in an attempt to calculate log(0) in
    # its cost function). To avoid this, all values of x that are higher than 709.7 are replaced with 709.7.
    x = np.minimum(x, 709.7)
    return np.exp(x)


def relu(x, a):
    y = x
    x[y == 0]


def leaky_relu(x, a):
    y = x
    x[y == 0]
