import numpy as np


def add_intercept(X):
    """Add intercept feature (all-ones)

    :param X: Features
    :return: Features with intercept feature
    """
    samples_count = X.shape[0]
    intercept_column = np.ones((samples_count, 1))
    return np.hstack((intercept_column, X))
