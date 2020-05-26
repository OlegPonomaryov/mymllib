"""Tests for the 'functions' module."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from mymllib.math.functions import sigmoid, tanh, relu, leaky_relu


def test_sigmoid():
    z = np.asarray([[100], [-100], [0]])
    expected = np.asarray([[1], [0], [0.5]])

    result = sigmoid(z)

    assert_allclose(result, expected, atol=5E-5)


def test_tanh():
    z = np.asarray([[100], [-100], [0]])
    expected = np.asarray([[1], [-1], [0]])

    result = tanh(z)

    assert_allclose(result, expected, atol=5E-5)


def test_relu():
    z = np.asarray([[-200, 0,     200],
                    [1,    1E-20, -1E-20]])
    expected = np.asarray([[0, 0,     200],
                           [1, 1E-20, 0]])

    result = relu(z)

    assert_array_equal(result, expected)


def test_leaky_relu():
    z = np.asarray([[-200, 0,     200],
                    [1,    1E-20, -1E-20]])
    expected = np.asarray([[-2, 0,     200],
                           [1, 1E-20, -1E-22]])

    result = leaky_relu(z)

    assert_allclose(result, expected, rtol=1E-15)
