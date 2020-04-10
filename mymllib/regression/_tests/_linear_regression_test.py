"""Tests for the '_linear_regression' module."""
import numpy as np
from numpy.testing import assert_allclose
from mymllib.regression import LinearRegression
from mymllib.tools import gradient
from mymllib.preprocessing import to_numpy


# A simple dataset for tests in which all samples lie precisely on a straight line to make it perfect for linear
# regression
X = [[3, 5],
     [6, 4],
     [3, 1],
     [-2, 5]]
y = [18, 19, 10, 13]

# An index from which the test part of the dataset starts
# (only data before this index can be used for training a model)
test_set_start = 3


def test_fit_predict__fit_and_predict_on_dataset__correct_predictions_returned():
    linear_regression = LinearRegression()
    linear_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = linear_regression.predict(X)

    assert_allclose(predictions, y)


def test_cost__analytical_cost_gradient_equal_to_numerical_one():
    X_np = to_numpy(X)
    coefs = np.ones(X_np.shape[1])

    linear_regression = LinearRegression(regularization_param=1)
    analytical_gradient = linear_regression._cost_gradient(coefs, X_np, y)
    numerical_gradient = gradient(coefs, linear_regression._cost, (X_np, y))

    assert_allclose(analytical_gradient, numerical_gradient)
