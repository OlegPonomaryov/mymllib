"""Functions that are used by metrics of multiple machine learning problems."""
from mymllib.preprocessing import to_numpy


def _check_input(y_actual, y_predicted):
    """Check that actual and predicted values have correct dimensionality and matching shapes and convert them to NumPy
    arrays.

    :param y_actual: Actual values
    :param y_predicted: Predicted values
    :return: NumPy arrays of actual and predicted values.
    """
    y_actual, y_predicted = to_numpy(y_actual, y_predicted)

    if y_actual.ndim != 1 or y_predicted.ndim != 1:
        raise ValueError("Actual and predicted values should be one-dimensional")

    if y_actual.shape != y_predicted.shape:
        raise ValueError("Shapes of actual and predicted values don't match")

    return y_actual, y_predicted
