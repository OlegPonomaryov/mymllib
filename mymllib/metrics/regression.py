import numpy as np
from ._common import _check_input


def mean_absolute_error(y_actual, y_predicted):
    """Calculate mean absolute error of a regression model.

    :param y_actual: Actual target values
    :param y_predicted: Predicted target values
    :return: Mean absolute error value
    """
    y_actual, y_predicted = _check_input(y_actual, y_predicted)
    return np.absolute(y_actual - y_predicted).mean()


def mean_absolute_percentage_error(y_actual, y_predicted):
    """Calculate mean absolute percentage error of a regression model.

    :param y_actual: Actual target values
    :param y_predicted: Predicted target values
    :return: Mean absolute percentage error value
    """
    y_actual, y_predicted = _check_input(y_actual, y_predicted)
    return np.absolute((y_actual - y_predicted) / y_actual).mean() * 100
