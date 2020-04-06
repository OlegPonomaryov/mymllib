"""Tests for the '_functions' module."""
import numpy as np
from preprocessing import add_polynomial
import pytest


def test_add_polynomial__matrix_and_floating_point_degree_passed__value_error_raised():
    A = np.asarray([[1, 2, 3],
                    [1, 2, 3]])
    polynomial_degree = 1.5

    with pytest.raises(ValueError):
        add_polynomial(A, polynomial_degree)


def test_add_polynomial__matrix_and_negative_degree_passed__value_error_raised():
    A = np.asarray([[1, 2, 3],
                    [1, 2, 3]])
    polynomial_degree = -1

    with pytest.raises(ValueError):
        add_polynomial(A, polynomial_degree)


def test_add_polynomial__matrix_and_degree_0_passed__value_error_raised():
    A = np.asarray([[1, 2, 3],
                    [1, 2, 3]])
    polynomial_degree = 0

    with pytest.raises(ValueError):
        add_polynomial(A, polynomial_degree)


def test_add_polynomial__matrix_and_degree_1_passed__same_matrix_returned():
    A = np.asarray([[1, 2, 3],
                    [1, 2, 3]])
    polynomial_degree = 1

    B = add_polynomial(A, polynomial_degree)

    assert np.array_equal(B, A)


def test_add_polynomial__matrix_and_degree_2_passed__up_to_2nd_degree_polynomial_matrix_returned():
    A = np.asarray([[1, 2, 3],
                    [1, 2, 3]])
    A_poly = np.asarray([[1, 2, 3, 1, 2, 3, 4, 6, 9],
                         [1, 2, 3, 1, 2, 3, 4, 6, 9]])
    polynomial_degree = 2

    B = add_polynomial(A, polynomial_degree)

    assert np.array_equal(B, A_poly)


def test_add_polynomial__matrix_and_degree_4_passed__up_to_4th_degree_polynomial_matrix_returned():
    A = np.asarray([[4, 6, 3],
                    [4, 6, 3]])
    polynomial_degree = 4
    A_poly = np.asarray(
        [[4, 6, 3, 16, 24, 12, 36, 18, 9, 64, 96, 48, 144, 72, 36, 216, 108, 54, 27, 256, 384, 192, 576, 288, 144, 864,
          432, 216, 108, 1296, 648, 324, 162, 81],
         [4, 6, 3, 16, 24, 12, 36, 18, 9, 64, 96, 48, 144, 72, 36, 216, 108, 54, 27, 256, 384, 192, 576, 288, 144, 864,
          432, 216, 108, 1296, 648, 324, 162, 81]])

    B = add_polynomial(A, polynomial_degree)

    assert np.array_equal(B, A_poly)
