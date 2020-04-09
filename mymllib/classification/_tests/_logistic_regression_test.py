"""Tests for the '_logistic_regression' module."""
from numpy.testing import assert_array_equal
from mymllib.classification import LogisticRegression

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

# A binary version of the y in which class 1 is class 2 from the multiclass version and class 0 stands for all other
# classes
y_bin = [0, 0, 1,  0, 0, 1,  0, 0, 1]

# An index from which the test part of the dataset starts
# (only data before this index can be used for training a model)
test_set_start = 6


def test_fit_predict__fit_and_predict_on_binary_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X[:test_set_start], y_bin[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y_bin)


def test_fit_predict__fit_and_predict_with_all_at_once_on_binary_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X[:test_set_start], y_bin[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y_bin)


def test_fit_predict__fit_and_predict_on_multiclass_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression(all_at_once=True)
    logistic_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y)


def test_fit_predict__fit_and_predict_with_all_at_on_multiclass_dataset__correct_predictions_returned():
    logistic_regression = LogisticRegression(all_at_once=True)
    logistic_regression.fit(X[:test_set_start], y[:test_set_start])
    predictions = logistic_regression.predict(X)

    assert_array_equal(predictions, y)
