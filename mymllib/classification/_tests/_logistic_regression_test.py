"""Tests for the '_logistic_regression' module."""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from mymllib.classification import LogisticRegression
from mymllib.preprocessing import DataScaler, to_numpy
from mymllib.tools import gradient


# A simple dataset for tests
# Classes:
#   0 - both x1 and x2 are high
#   1 - both x1 and x2 are low
#   2 - x1 is high, x2 is low
X = [[24, 32],
     [3, 0],
     [19, 1],

     [17, 28],
     [0, 5],
     [27, 5],

     [20, 30],
     [2, 3],
     [22, 3]]
y = [0, 1, 2,  0, 1, 2,  0, 1, 2]

# Due to significant digits limitation of floating-point variables an output of the logistic function for very large or
# very small arguments is rounded, so altering such an argument a little bit won't change the result of the function,
# making numerical gradient calculation impossible. This can be avoided by scaling X and therefore decreasing absolute
# values of its elements.
X_scaled = DataScaler().fit(X).scale(X)

# A one-hot version of the y
y_one_hot = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],

             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],

             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]

# A binary version of the y with respect to the class 2
y_bin = [0, 0, 1,  0, 0, 1,  0, 0, 1]

# An index from which the test part of the dataset starts
# (only data before this index can be used for training a model)
test_set_start = 6


def test_fit_predict__fit_and_predict_on_binary_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression(all_at_once=False)
    logistic_regression.fit(X[:test_set_start], y_bin[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y_bin)


def test_fit_predict__fit_and_predict_with_all_at_once_on_binary_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression(all_at_once=True)
    logistic_regression.fit(X[:test_set_start], y_bin[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y_bin)


def test_fit_predict__fit_and_predict_on_multiclass_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression(all_at_once=False)
    logistic_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y)


def test_fit_predict__fit_and_predict_with_all_at_once_on_multiclass_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression(all_at_once=True)
    logistic_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y)


# This test imitates calculating cost and its gradient for binary classification problems or multiclass ones when
# all_at_once is set to False and coefficients for each binary subproblems are being optimized separately
def test_cost__binary_y_and_1D_coefs_passed__analytical_cost_gradient_equal_to_numerical_one():
    y_np = to_numpy(y_bin)
    coefs = np.ones(X_scaled.shape[1])

    logistic_regression = LogisticRegression(regularization_param=1)
    analytical_gradient = logistic_regression._cost_gradient(coefs, X_scaled, y_np)
    numerical_gradient = gradient(coefs, logistic_regression._cost, (X_scaled, y_np))

    assert_allclose(analytical_gradient, numerical_gradient)


# This test imitates calculating cost and its gradient for multiclass problems when all_at_once is set to True and
# coefficients for each binary subproblems are optimized all at once as a matrix
def test_cost__one_hot_y_and_2D_coefs_passed__analytical_cost_gradient_equal_to_numerical_one():
    y_np = to_numpy(y_one_hot)
    coefs = np.ones((X_scaled.shape[1], np.shape(y_np)[1]))

    logistic_regression = LogisticRegression(regularization_param=1)
    analytical_gradient = logistic_regression._cost_gradient(coefs, X_scaled, y_np)
    numerical_gradient = gradient(coefs, logistic_regression._cost, (X_scaled, y_np))

    assert_allclose(analytical_gradient, numerical_gradient)
