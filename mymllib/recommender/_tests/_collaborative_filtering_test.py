"""Tests for the CollaborativeFiltering class."""
import numpy as np

import pytest
from numpy.testing import assert_array_equal, assert_allclose

from mymllib.recommender import CollaborativeFiltering
from mymllib.tools import glorot_init
from mymllib.optimization import unroll
from mymllib.math.tools import gradient
from mymllib.preprocessing import to_numpy


# Simple movies ratings dataset from the Machine Learning course by Andrew Ng
X_train = [["Alice", "Love at last"],
           ["Alice", "Romance forever"],
           ["Alice", "Nonstop car chases"],
           ["Alice", "Swords vs. karate"],
           ["Bob", "Love at last"],
           ["Bob", "Cute puppies of love"],
           ["Bob", "Nonstop car chases"],
           ["Bob", "Swords vs. karate"],
           ["Carol", "Love at last"],
           ["Carol", "Cute puppies of love"],
           ["Carol", "Nonstop car chases"],
           ["Carol", "Swords vs. karate"],
           ["Dave", "Love at last"],
           ["Dave", "Romance forever"],
           ["Dave", "Nonstop car chases"]]
y_train = [5, 5, 0, 0, 5, 4, 0, 0, 0, 0, 5, 5, 0, 0, 4]

# Expected ratings matrix
Y_expected = [[np.nan, 4,      0,      np.nan],
              [5,      5,      0,      0],
              [0,      0,      5,      4],
              [5,      np.nan, np.nan, 0],
              [0,      0,      5,      np.nan]]

# Expected list of unique users
users_expected = ["Alice", "Bob", "Carol", "Dave"]

# Expected list of unique movies
movies_expected = ["Cute puppies of love", "Love at last", "Nonstop car chases", "Romance forever", "Swords vs. karate"]

# Test data that include all unrated movies for each user
X_test = [["Alice", "Cute puppies of love"],
          ["Bob", "Romance forever"],
          ["Carol", "Romance forever"],
          ["Dave", "Cute puppies of love"],
          ["Dave", "Swords vs. karate"],
          ["Dave", "Deadly space rabbits"],  # Movie is new, rating should be 0
          ["Anthony", "Swords vs. karate"],  # User is new, rating should be the average rating of the movie
          ["Anthony", "Deadly space rabbits"]]  # Both user and movie are new, rating should be 0

# Expected ratings predictions
y_test_expected = [4, 5, 0, 0, 4, 0, 5/3, 0]


def test_build_ratings_matrix():
    collaborative_filtering = CollaborativeFiltering(7)

    Y, users, movies = collaborative_filtering._build_ratings_matrix(to_numpy(X_train), to_numpy((y_train)))

    assert_array_equal(Y, Y_expected)
    assert_array_equal(users, users_expected)
    assert_array_equal(movies, movies_expected)


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
@pytest.mark.parametrize("Y", [Y_expected])
@pytest.mark.parametrize("regularization_param", [0, 1, 10])
def test_cost_gradient(features_count, regularization_param, Y):
    Y = to_numpy(Y)
    users_count = Y.shape[1]
    items_count = Y.shape[0]
    params = unroll(glorot_init(((users_count, features_count), (items_count, features_count))))

    collaborative_filtering = CollaborativeFiltering(features_count, regularization_param)
    collaborative_filtering._users_count = users_count
    collaborative_filtering._items_count = items_count

    analytical_gradient = collaborative_filtering._cost_gradient(params, Y)
    numerical_gradient = gradient(params, collaborative_filtering._cost, (Y,))

    assert_allclose(analytical_gradient, numerical_gradient, rtol=1E-4, atol=1E-4)


def test_fit_predict():
    collaborative_filtering = CollaborativeFiltering(2, 0)

    collaborative_filtering.fit(X_train, y_train)
    y_test = collaborative_filtering.predict(X_test)

    assert_allclose(y_test, y_test_expected, rtol=1E-4, atol=1E-4)


@pytest.mark.parametrize("item, count, expected", [
    ("Cute puppies of love", 2, {"Love at last", "Romance forever"}),
    ("Nonstop car chases", 1, {"Swords vs. karate"})])
def test_find_similar_items(item, count, expected):
    collaborative_filtering = CollaborativeFiltering(2, 0)
    collaborative_filtering.fit(X_train, y_train)

    similar_items = collaborative_filtering.find_similar_items(item, count)

    assert set(similar_items) == expected


# This test verifies that the function can handle requested items count that is greater then total items count and also
# checks that similar items will appear in the list earlier
def test_find_similar_items__check_order():
    collaborative_filtering = CollaborativeFiltering(2, 0)
    collaborative_filtering.fit(X_train, y_train)

    similar_items = collaborative_filtering.find_similar_items("Cute puppies of love", 100)

    assert set(similar_items[:2]) == {"Love at last", "Romance forever"}
    assert set(similar_items[2:]) == {"Nonstop car chases", "Swords vs. karate"}
