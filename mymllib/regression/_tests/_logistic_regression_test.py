"""Tests for the LogisticRegression class."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from mymllib.regression import LogisticRegression
from mymllib.preprocessing import DataScaler, to_numpy
from mymllib.math.tools import gradient
from mymllib._test_data.classification import X, y, y_text, y_one_hot, y_bin, test_set_start


@pytest.mark.parametrize("X, y", [
    (np.ones(20), np.ones(20)),
    (np.ones((20, 5, 3)), np.ones(20)),
    (np.ones((20, 5)), np.ones((20, 5))),
    (np.ones((20, 5)), np.ones(30))])
def test_fit__invalid_input_shapes(X, y):
    logistic_regression = LogisticRegression(optimizer=None)

    with pytest.raises(ValueError):
        logistic_regression.fit(X, y)


@pytest.mark.parametrize("X_test", [
    np.ones(10), np.ones((10, 2, 1)), np.ones((10, 1))])
def test_predict__invalid_input_shapes(X_test):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X[:test_set_start], y[:test_set_start])

    with pytest.raises(BaseException):
        logistic_regression.predict(X_test)


@pytest.mark.parametrize("y", [y_bin, y, y_text])
@pytest.mark.parametrize("all_at_once", [True, False])
@pytest.mark.parametrize("regularization_param", [0, 1])
def test_fit_predict(y, all_at_once, regularization_param):
    logistic_regression = LogisticRegression(all_at_once=all_at_once, regularization_param=regularization_param)

    logistic_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y)


@pytest.mark.parametrize("y, params", [
    (y_bin, np.ones(np.shape(X)[1])),
    (y_one_hot, np.ones((np.shape(X)[1], np.shape(y_one_hot)[1])))
])
@pytest.mark.parametrize("regularization_param", [0, 1])
def test_cost_gradient(y, params, regularization_param):
    # Due to significant digits limitation of floating-point variables an output of the logistic function for very large
    # or very small arguments is rounded, so altering such an argument a little bit won't change the result of the
    # function, making numerical gradient calculation impossible. This can be avoided by scaling X and therefore
    # decreasing absolute values of its elements.
    X_scaled = DataScaler().fit(X).scale(X)

    y_np = to_numpy(y)
    logistic_regression = LogisticRegression(regularization_param=regularization_param)

    analytical_gradient = logistic_regression._cost_gradient(params, X_scaled, y_np)
    numerical_gradient = gradient(params, logistic_regression._cost, (X_scaled, y_np))

    assert_allclose(analytical_gradient, numerical_gradient)
