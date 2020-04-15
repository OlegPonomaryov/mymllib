"""Tests for the 'functions' module."""
import numpy as np
from numpy.testing import assert_allclose
from mymllib.math.functions import sigmoid


def test_sigmoid():
    # X and theta are selected so that their product will be equal to [-4.6, 0, 4.6]
    X = np.asarray([[4.6, 0, 0], [0, 22, 0], [0, 0, 2.3]])
    theta = np.asarray([-1, 0, 2])
    expected = [0.01, 0.5, 0.99]

    result = sigmoid(X, theta)

    assert_allclose(result, expected, atol=5E-5)


# Test that for a very large argument the sigmoid function returns a value close to 0, but not exactly equal to it (to
# avoid errors when calculating log(sigmoid()) in the logistic regression cost function)
def test_sigmoid__very_small_argument__result_greater_than_0():
    # X and theta are selected so that their product will be equal to [-4.6, 0, 4.6]
    X = np.asarray([-1000000000])
    theta = np.asarray([1000000000])

    result = sigmoid(X, theta)

    assert_allclose(result, 0, atol=1E-15)
    assert result > 0


# Test that for a very large argument the sigmoid function returns a value close to 1, but not exactly equal to it (to
# avoid errors when calculating log(1 - sigmoid()) in the logistic regression cost function)
def test_sigmoid__very_large_argument__result_less_than_1():
    # X and theta are selected so that their product will be equal to [-4.6, 0, 4.6]
    X = np.asarray([1000000000])
    theta = np.asarray([1000000000])

    result = sigmoid(X, theta)

    assert_allclose(result, 1, atol=1E-15)
    assert result < 1

