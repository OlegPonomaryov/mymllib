"""Tests for the '_data_scaler' module."""
import numpy as np
from preprocessing import DataScaler


def test_fit_scale__fit_on_one_matrix_and_scale_another__correctly_scaled_matrix_returned():
    A = [[0, -8, 5],
         [4, -2, 7]]
    B = [[5, 7, 4],
         [2, 12, 1]]
    B_scaled = [[1.5, 4, -2],
                [0, 17/3, -5]]

    data_scaler = DataScaler()
    data_scaler.fit(A)
    C = data_scaler.scale(B)

    assert np.array_equal(C, B_scaled)