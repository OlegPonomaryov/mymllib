"""Tests for the DataScaler class."""
from numpy.testing import assert_array_equal
from mymllib.preprocessing import DataScaler


def test_fit_scale():
    A = [[0, -8, 5],
         [4, -2, 7]]
    B = [[5, 7, 4],
         [2, 12, 1]]
    B_scaled = [[1.5, 4, -2],
                [0, 17/3, -5]]

    data_scaler = DataScaler()
    data_scaler.fit(A)
    C = data_scaler.scale(B)

    assert_array_equal(C, B_scaled)
