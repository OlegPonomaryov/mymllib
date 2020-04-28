"""Tests for the 'tools' module."""
from math import exp, log
from numpy.testing import assert_allclose
from mymllib.preprocessing import to_numpy
from mymllib.math.tools import gradient


def f(x):
    return exp(x[0]) + 3*x[1]**2 - log(x[2])


def grad_f(x):
    return [exp(x[0]), 6*x[1], -1/x[2]]


def test_gradient():
    x = to_numpy([5, -9, 12])

    numerical_grad = gradient(x, f)

    analytical_grad = grad_f(x)
    assert_allclose(numerical_grad, analytical_grad)
