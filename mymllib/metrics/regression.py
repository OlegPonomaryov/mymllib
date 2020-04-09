import numpy as np


def mean_absolute_error(y_true, y_predicted):
    """Calculate mean absolute error of a regression model.

    :param y_true: Actual target values
    :param y_predicted: Predicted target values
    :return: Mean absolute error value
    """
    return np.absolute(y_true - y_predicted).mean()


def mean_absolute_percentage_error(y_true, y_predicted):
    """Calculate mean absolute percentage error of a regression model.

    :param y_true: Actual target values
    :param y_predicted: Predicted target values
    :return: Mean absolute percentage error value
    """
    return np.absolute((y_true - y_predicted) / y_true).mean() * 100
