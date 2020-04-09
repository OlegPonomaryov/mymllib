"""Tests for the '_linear_regression' module."""
from numpy.testing import assert_allclose
from mymllib.regression import LinearRegression


def test_fit_predict__fit_and_predict_on_dataset__correct_predictions_returned():
    X = [[3, 5],
         [6, 4],
         [3, 1],
         [-2, 5]]
    y = [18, 19, 10, 13]
    test_set_length = 3

    linear_regression = LinearRegression()
    linear_regression.fit(X[:test_set_length], y[:test_set_length])
    predictions = linear_regression.predict(X)

    assert_allclose(predictions, y)
