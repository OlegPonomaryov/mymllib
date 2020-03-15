import numpy as np


class LinearRegression:
    def __init__(self):
        self._coefs = None

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X = LinearRegression._add_intercept(X)
        X_T = X.transpose()
        self._coefs = np.linalg.pinv(X_T @ X) @ X_T @ y

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
