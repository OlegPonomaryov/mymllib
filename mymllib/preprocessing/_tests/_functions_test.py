"""Tests for the '_functions' module."""
import pytest
from numpy import ndarray
from numpy.testing import assert_array_equal
from mymllib.preprocessing import to_numpy, add_intercept, add_polynomial, one_hot


A = [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]


def test_to_numpy__one_arg_passed():
    A_numpy = to_numpy(A)

    assert isinstance(A_numpy, ndarray)


def test_to_numpy__two_args_passed():
    A_numpy, B_numpy = to_numpy(A, [1, 2, 3])

    assert isinstance(A_numpy, ndarray)
    assert isinstance(B_numpy, ndarray)


def test_add_intercept():
    A_intercept = [[1, 1, 2, 3],
                   [1, 1, 2, 3],
                   [1, 1, 2, 3]]

    B = add_intercept(A)

    assert_array_equal(B, A_intercept)


@pytest.mark.parametrize("polynomial_degree", [1.5, -1, 0])
def test_add_polynomial__invalid_degree(polynomial_degree):
    with pytest.raises(ValueError):
        add_polynomial(A, polynomial_degree)


@pytest.mark.parametrize("polynomial_degree, expectation", [
    (1, A),
    (2, [[1, 2, 3, 1, 2, 3, 4, 6, 9],
         [1, 2, 3, 1, 2, 3, 4, 6, 9],
         [1, 2, 3, 1, 2, 3, 4, 6, 9]]),
    (3, [[1, 2, 3, 1, 2, 3, 4, 6, 9, 1, 2, 3, 4, 6, 9, 8, 12, 18, 27],
         [1, 2, 3, 1, 2, 3, 4, 6, 9, 1, 2, 3, 4, 6, 9, 8, 12, 18, 27],
         [1, 2, 3, 1, 2, 3, 4, 6, 9, 1, 2, 3, 4, 6, 9, 8, 12, 18, 27]])])
def test_add_polynomial(polynomial_degree, expectation):
    result = add_polynomial(A, polynomial_degree)

    assert_array_equal(result, expectation)


y_text_bin = ["A", "B", "A", "A", "B", "A", "A", "B", "A"]
expected_labels_bin = ["A", "B"]
expected_y_one_hot_bin = [[0],
                          [1],
                          [0],

                          [0],
                          [1],
                          [0],

                          [0],
                          [1],
                          [0]]
y_text = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
expected_labels = ["A", "B", "C"]
expected_y_one_hot = [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],

                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],

                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]


@pytest.mark.parametrize("y, expected_labels, expected_y_one_hot", [
    (["A", "B", "A", "A", "B", "A", "A", "B", "A"], ["A", "B"], [[0], [1], [0], [0], [1], [0], [0], [1], [0]]),
    (["A", "B", "C", "A", "B", "C", "A", "B", "C"], ["A", "B", "C"],
     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
])
def test_one_hot(y, expected_labels, expected_y_one_hot):
    labels, y_one_hot = one_hot(to_numpy(y))

    assert_array_equal(labels, expected_labels)
    assert_array_equal(y_one_hot, expected_y_one_hot)
