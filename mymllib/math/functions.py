"""Implementations of different mathematical functions."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + safe_exp(-x))


def log_loss(predicted, actual):
    """Return logistic (sigmoid) loss function value."""
    return -(actual * safe_log(predicted) + (1 - actual) * safe_log(1 - predicted))


def log_cost(predicted, actual):
    """Return logistic (sigmoid) cost function value (without regularization)."""
    return np.sum(log_loss(predicted, actual)) / predicted.shape[0]


def softmax(x):
    t = safe_exp(x)
    h = t / t.sum(axis=1, keepdims=True)
    return h


def softmax_loss(predicted, actual):
    """Return softmax loss function value."""
    return -(actual * safe_log(predicted))


def softmax_cost(predicted, actual):
    """Return softmax cost function value (without regularization)."""
    return np.sum(softmax_loss(predicted, actual)) / predicted.shape[0]


def tanh(x):
    exp_plus, exp_minus = safe_exp(x), safe_exp(-x)
    return (exp_plus - exp_minus) / (exp_plus + exp_minus)


def safe_log(x):
    """Avoids calculating logarithm of 0 by adding a small value to its argument."""
    return np.log(x + 1E-15)


def safe_exp(x):
    """Avoids calculating exponent for arguments too big to store the result in float variable by replacing these
        arguments with roughly highest acceptable value."""
    x = np.minimum(x, 709.7)
    return np.exp(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    y = x.astype(float, copy=True)
    y[y < 0] *= 0.01
    return y
