import numpy as np


class LinearRegression:
    """Linear regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regression is used)
    """

    def __init__(self, regularization_param=0):
        assert regularization_param >= 0, "Regularization parameter must be >= 0, but was negative."
        self._coefs = None
        self._regularization_param = regularization_param

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X = LinearRegression._add_intercept(X)
        X_T = X.transpose()

        features_count = X_T.shape[0]
        regularization_matrix = self._regularization_param*np.identity(features_count)
        regularization_matrix[0, 0] = 0

        self._coefs = np.linalg.pinv(X_T @ X + regularization_matrix) @ X_T @ y

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        return LinearRegression._add_intercept(X)@self._coefs

    @staticmethod
    def _add_intercept(X):
        """Add intercept feature (all-ones)

        :param X: Features
        :return: Features with intercept feature
        """
        samples_count = X.shape[0]
        intercept_column = np.ones((samples_count, 1))
        return np.hstack((intercept_column, X))
