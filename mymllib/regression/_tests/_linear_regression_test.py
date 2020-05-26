"""Tests for the LinearRegression class."""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from mymllib._test_data.regression import X_train, y_train, X_test, y_test
from mymllib.regression import LinearRegression
from mymllib.optimization import LBFGSB
from mymllib.math.tools import gradient
from mymllib.preprocessing import to_numpy


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
    linear_regression.fit(X_train, y_train)

    with pytest.raises(ValueError):
        linear_regression.predict(X_test)


@pytest.mark.parametrize("optimizer", [None, LBFGSB()])
@pytest.mark.parametrize("regularization_param", [0, 0.01])
def test_fit_predict(optimizer, regularization_param):
    linear_regression = LinearRegression(optimizer=optimizer, regularization_param=regularization_param)

    linear_regression.fit(X_train, y_train)
    predictions = linear_regression.predict(X_test)

    assert_allclose(predictions, y_test, rtol=1e-3)


@pytest.mark.parametrize("regularization_param", [0, 1])
def test_cost_gradient(regularization_param):
    X = to_numpy(X_train)
    params = np.ones(X.shape[1])

    linear_regression = LinearRegression(regularization_param=regularization_param)
    analytical_gradient = linear_regression._cost_gradient(params, X, y_train)
    numerical_gradient = gradient(params, linear_regression._cost, (X, y_train))

    assert_allclose(analytical_gradient, numerical_gradient)
