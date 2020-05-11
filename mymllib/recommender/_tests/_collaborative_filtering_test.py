"""Tests for the CollaborativeFiltering class."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from mymllib.recommender import CollaborativeFiltering
from mymllib.tools import glorot_init
from mymllib.optimization import unroll
from mymllib.math.tools import gradient


# Simple ratings matrix
Y = np.asarray(
    [[5,      5,      0,      0],
     [5,      np.nan, np.nan, 0],
     [np.nan, 4,      0,      np.nan],
     [0,      0,      5,      4],
     [0,      0,      5,      np.nan]])

# Simple test data to predict ratings
X = np.asarray([[0, 2],
                [1, 1],
                [2, 1],
                [3, 2],
                [3, 4]])

# Expected ratings prediction
expected = [4, 5, 0, 0, 4]


@pytest.mark.parametrize("predicted", [
    [[3, 1],
     [2, -5]]
])
@pytest.mark.parametrize("actual, expected", [
    ([[3, 1],
      [2, -5]],
     [[0, 0],
      [0, 0]]),

    ([[1, 0],
      [3, -3]],
     [[2, 1],
      [-1, -2]]),

    ([[1, np.nan],
      [np.nan, -3]],
     [[2, 0],
      [0, -2]])
])
def test_error(predicted, actual, expected):
    collaborative_filtering = CollaborativeFiltering(7)

    error = collaborative_filtering._error(np.asarray(predicted), np.asarray(actual))

    assert_array_equal(error, np.asarray(expected))


@pytest.mark.parametrize("features_count", [3, 7, 20])
@pytest.mark.parametrize("Y", [Y])
@pytest.mark.parametrize("regularization_param", [0, 1, 10])
def test_cost_gradient(features_count, regularization_param, Y):
    users_count = Y.shape[1]
    items_count = Y.shape[0]
    params = unroll(glorot_init(((users_count, features_count), (items_count, features_count))))

    collaborative_filtering = CollaborativeFiltering(features_count, regularization_param)
    collaborative_filtering._users_count = users_count
    collaborative_filtering._items_count = items_count

    analytical_gradient = collaborative_filtering._cost_gradient(params, Y)
    numerical_gradient = gradient(params, collaborative_filtering._cost, (Y,))

    assert_allclose(analytical_gradient, numerical_gradient, rtol=1E-5)


@pytest.mark.parametrize("features_count, regularization_param", [(2, 0)])
@pytest.mark.parametrize("Y, X, expected", [(Y, X, expected)])
def test_fit_predict(features_count, regularization_param, Y, X, expected):
    collaborative_filtering = CollaborativeFiltering(features_count, regularization_param)

    collaborative_filtering.fit(Y)
    y = collaborative_filtering.predict(X)

    assert_allclose(y, expected, rtol=1E-4, atol=1E-4)


