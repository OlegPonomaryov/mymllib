import numpy as np
from ..preprocessing import add_intercept
from ._base_regression import BaseRegression


class LinearRegression(BaseRegression):
    """Linear regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is used)
    :param use_gradient_descent: Whether to use gradient descent instead of normal equation to fit the model
    :param learning_rate: Initial learning rate of gradient descent (used only if use_gradient_descent set to True, can be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent (used only if use_gradient_descent set to True)
    :param max_iterations: Maximum iterations count of gradient descent (used only if use_gradient_descent set to True)
    """

    def __init__(self, regularization_param=0, use_gradient_descent=False,
                 learning_rate=1, accuracy=1E-5, max_iterations=10000):
        super().__init__(regularization_param, learning_rate, accuracy, max_iterations)
        self._use_gradient_descent = use_gradient_descent

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X = add_intercept(X)
        if self._use_gradient_descent:
            super().fit(X, y)
        else:
            self._coefs = self._normal_equation(X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        return super().predict(add_intercept(X))

    def _hypothesis(self, X, coefs):
        return X@coefs

    def _cost(self, coefs, X, y):
        return (((self._hypothesis(X, coefs) - y)**2).sum() +
                self._regularization_param*(coefs[1:]**2).sum()) / X.shape[0]

    def _normal_equation(self, X, y):
        features_count = X.T.shape[0]
        regularization_matrix = self._regularization_param*np.identity(features_count)
        regularization_matrix[0, 0] = 0
        return np.linalg.pinv(X.T @ X + regularization_matrix) @ X.T @ y
