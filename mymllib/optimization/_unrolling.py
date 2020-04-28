"""Tools for unrolling arrays into a single one-dimensional array and restoring them back."""
import numpy as np


ORDER = 'C'  # An order to use for unrolling


def unroll(arrays):
    """Unroll a sequence of arrays into a single one-dimensional array.

    :param arrays: Sequence of NumPy arrays
    :return: One-dimensional NumPy array
    """
    flattened = tuple(array.flatten(order=ORDER) for array in arrays)
    return np.hstack(flattened)


def undo_unroll(source_array, shapes):
    """Undo unrolling by converting a one-dimensional array into a sequence of arrays with specified shapes.

    :param source_array: One-dimensional NumPy array
    :param shapes: Shapes of arrays to return (sequences of integers)
    :return: Sequence of NumPy arrays
    """
    if source_array.ndim != 1:
        raise ValueError("Passed source array isn't one-dimensional")

    expected_elements_count = sum(np.prod(shape) for shape in shapes)
    if expected_elements_count != source_array.size:
        raise ValueError("Size of the source array doesn't match passed shapes")

    start = 0
    result = list()
    for shape in shapes:
        result.append(source_array[start:start + np.prod(shape)].reshape(shape, order=ORDER))
        start += np.prod(shape)

    return result
