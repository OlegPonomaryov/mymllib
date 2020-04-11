"""Tests for the LogisticRegression class."""
import pytest
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

y_text = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]

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

# An index from which the test part of the dataset starts (only data before this index can be used for training a model)
test_set_start = 6


@pytest.mark.parametrize("y", [y_bin, y, y_text])
@pytest.mark.parametrize("all_at_once", [True, False])
@pytest.mark.parametrize("regularization_param", [0, 1])
def test_fit_predict(y, all_at_once, regularization_param):
    logistic_regression = LogisticRegression(all_at_once=all_at_once, regularization_param=regularization_param)

    logistic_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y)


@pytest.mark.parametrize("y, coefs", [
    (y_bin, np.ones(np.shape(X)[1])),
    (y_one_hot, np.ones((np.shape(X)[1], np.shape(y_one_hot)[1])))
])
@pytest.mark.parametrize("regularization_param", [0, 1])
def test_cost_gradient(y, coefs, regularization_param):
    # Due to significant digits limitation of floating-point variables an output of the logistic function for very large
    # or very small arguments is rounded, so altering such an argument a little bit won't change the result of the
    # function, making numerical gradient calculation impossible. This can be avoided by scaling X and therefore
    # decreasing absolute values of its elements.
    X_scaled = DataScaler().fit(X).scale(X)

    y_np = to_numpy(y)
    logistic_regression = LogisticRegression(regularization_param=regularization_param)

    analytical_gradient = logistic_regression._cost_gradient(coefs, X_scaled, y_np)
    numerical_gradient = gradient(coefs, logistic_regression._cost, (X_scaled, y_np))

    assert_allclose(analytical_gradient, numerical_gradient)


def test_one_hot():
    logistic_regression = LogisticRegression()

    labels, one_hot = logistic_regression._one_hot(to_numpy(y_text))

    assert_array_equal(labels, ["A", "B", "C"])
    assert_array_equal(one_hot, y_one_hot)
