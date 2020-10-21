"""Tests for the DataScaler class."""
import pytest
from numpy.testing import assert_array_equal
from mymllib.preprocessing import DataScaler

# The last column of A has 0 standard deviation which should be ignored during scaling to avoid 'inf' values
A = [[0, -8, 5, -3],
     [4, -2, 7, -3]]

B = [[5, 7, 4, 4],
     [2, 12, 1, 1]]

B_mean_scaled = [[3, 12, -2, 7],
                 [0, 17, -5, 4]]

B_std_scaled = [[2.5, 7/3, 4, 4],
                [1,   4,   1, 1]]

B_scaled = [[1.5, 4,    -2, 7],
            [0,   17/3, -5, 4]]


@pytest.mark.parametrize("A, B, by_mean, by_std, expected", [
    (A, B, True, False, B_mean_scaled),
    (A, B, False, True, B_std_scaled),
    (A, B, True, True, B_scaled)
])
def test_fit_scale(A, B, by_mean, by_std, expected):

    data_scaler = DataScaler()

    data_scaler.fit(A)
    scaled = data_scaler.scale(B, by_mean=by_mean, by_std=by_std)

    assert_array_equal(scaled, expected)
