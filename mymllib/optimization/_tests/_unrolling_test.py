"""Tests for the '_unrolling' module."""
import pytest
from numpy.testing import assert_array_equal
import numpy as np
from mymllib.optimization import unroll, undo_unroll


a1, a2, a3 = np.arange(10), np.arange(22, 38), np.ones(12)
# Test matrices
matrices = (a1.reshape((2, 5), order='C'), a2.reshape((4, 4), order='C'), a3.reshape((4, 3), order='C'))
# A one-dimensional array that is expected after unrolling the test matrices
unrolled_matrices = np.hstack((a1, a2, a3))


@pytest.mark.parametrize("arrays, expected", [
    (matrices, unrolled_matrices),
    (np.arange(10), np.arange(10))
])
def test_unroll(arrays, expected):
    result = unroll(arrays)

    assert_array_equal(result, expected)


def test_undo_unroll__source_array_not_1D():
    with pytest.raises(ValueError):
        undo_unroll(np.ones((9, 1)), ((3, 3),))


@pytest.mark.parametrize("source_array, shapes", [
    (np.ones(100), ((5, 7), (3, 6))),  # Source array contains more elements than specified by shapes
    (np.ones(10), ((4, 3),)),  # Source array contains less elements than specified by shapes
])
def test_undo_unroll__invalid_shapes(source_array, shapes):
    with pytest.raises(ValueError):
        undo_unroll(source_array, shapes)


@pytest.mark.parametrize("source_array, shapes, expected", [
    (unrolled_matrices, [matrix.shape for matrix in matrices], matrices),
    (np.arange(10), [(10,)], (np.arange(10),))
])
def test_undo_unroll(source_array, shapes, expected):
    result = undo_unroll(source_array, shapes)

    for i in range(len(expected)):
        assert_array_equal(result[i], expected[i])
