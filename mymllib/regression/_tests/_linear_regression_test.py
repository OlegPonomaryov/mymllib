"""Tests for the LinearRegression class."""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from mymllib.regression import LinearRegression
from mymllib.optimization import LBFGSB
from mymllib.math.tools import gradient
from mymllib.preprocessing import to_numpy


# A simple dataset for tests in which all samples lie precisely on a straight line to make it perfect for linear
# regression
X = [[3, 5],
     [6, 4],
     [3, 1],
     [-2, 5]]
y = [18, 19, 10, 13]

# An index from which the test part of the dataset starts (only data before this index can be used for training a model)
test_set_start = 3


@pytest.mark.parametrize("X, y", [
    (np.ones(20), np.ones(20)),
    (np.ones((20, 5, 3)), np.ones(20)),
    (np.ones((20, 5)), np.ones((20, 5))),
    (np.ones((20, 5)), np.ones(30))])
def test_fit__invalid_input_shapes(X, y):
    linear_regression = LinearRegression()

    with pytest.raises(ValueError):
        linear_regression.fit(X, y)


@pytest.mark.parametrize("X_test", [
    np.ones(10), np.ones((10, 2, 1)), np.ones((10, 1))])
def test_predict__invalid_input_shapes(X_test):
    linear_regression = LinearRegression()
    linear_regression.fit(X[:test_set_start], y[:test_set_start])

    with pytest.raises(ValueError):
        linear_regression.predict(X_test)


@pytest.mark.parametrize("optimizer", [None, LBFGSB()])
# Higher regularization parameter will worsen precision because all features are actually 'useful' and don't need to be
# regularized
@pytest.mark.parametrize("regularization_param", [0, 0.00001])
def test_fit_predict(optimizer, regularization_param):
    linear_regression = LinearRegression(optimizer=optimizer, regularization_param=regularization_param)

    linear_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = linear_regression.predict(X)

    assert_allclose(predictions, y, rtol=1e-06)


@pytest.mark.parametrize("regularization_param", [0, 1])
def test_cost_gradient(regularization_param):
    X_np = to_numpy(X)
    coefs = np.ones(X_np.shape[1])

    linear_regression = LinearRegression(regularization_param=regularization_param)
    analytical_gradient = linear_regression._cost_gradient(coefs, X_np, y)
    numerical_gradient = gradient(coefs, linear_regression._cost, (X_np, y))

    assert_allclose(analytical_gradient, numerical_gradient)
