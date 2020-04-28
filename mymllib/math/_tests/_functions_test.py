"""Tests for the 'functions' module."""
import numpy as np
from numpy.testing import assert_allclose
from mymllib.math.functions import sigmoid


def test_sigmoid():
    z = np.asarray([[-4.6, 0, 4.6],
                    [100, -100, 0]])
    expected = [[0.01, 0.5, 0.99],
                [1,    0,   0.5]]

    result = sigmoid(z)

    assert_allclose(result, expected, atol=5E-5)


# Test that for a very large argument the sigmoid function returns a value close to 0, but not exactly equal to it (to
# avoid errors when calculating log(sigmoid()) in the logistic regression cost function)
def test_sigmoid__very_small_argument__result_greater_than_0():
    z = np.asarray([[-1E9, -1E9],
                    [-1E9, -1E9]])

    result = sigmoid(z)

    assert_allclose(result, 0, atol=1E-15)
    assert np.count_nonzero(result <= 0) == 0


# Test that for a very large argument the sigmoid function returns a value close to 1, but not exactly equal to it (to
# avoid errors when calculating log(1 - sigmoid()) in the logistic regression cost function)
def test_sigmoid__very_large_argupment__result_less_than_1():
    z = np.asarray([[1E9, 1E9],
                    [1E9, 1E9]])

    result = sigmoid(z)

    assert_allclose(result, 1, atol=1E-15)
    assert np.count_nonzero(result >= 1) == 0
