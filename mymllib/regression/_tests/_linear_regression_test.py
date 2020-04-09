"""Tests for the '_linear_regression' module."""
from numpy.testing import assert_allclose
from mymllib.regression import LinearRegression

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
