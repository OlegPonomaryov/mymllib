import numpy as np
from ..preprocessing import add_intercept
from ..regression._linear_regression import BaseRegression


class LogisticRegression(BaseRegression):
    """Logistic regression implementation.

    :param threshold: Threshold for classification
    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is used)
    :param learning_rate: Initial learning rate of gradient descent (can be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent
    :param max_iterations: Maximum iterations count of gradient descent
    """

    def __init__(self, threshold=0.5, regularization_param=0, learning_rate=1, accuracy=1E-5, max_iterations=10000):
        super().__init__(regularization_param, learning_rate, accuracy, max_iterations)
        self._threshold = threshold

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X = add_intercept(X)
        super().fit(X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        return (super().predict(add_intercept(X)) >= self._threshold) * 1

    def _hypothesis(self, X, coefs):
        z = X @ coefs

        # A value of exp(n) exceeds a capacity of double-precision floating-point variables if n is higher than
        # approximately 709.7. For np.exp() this results in warning message and inf return value, which also makes the
        # hypothesis to return 0 resulting in an attempt to calculate log(0) in the cost function. To avoid this, all
        # values from z that are lower than -709.7 (because z is used with '-' in np.exp()) are replaced with -709.7.
        z = np.maximum(z, -709.7)

        h = 1 / (1 + np.exp(-z))

        # Values that are very close to 1 (like 0.9999999999999999999999) cannot be stored in double-precision floating-
        # point variables due to their significant digits limitation and are rounded to 1. But returning exactly 1 will
        # result in an attempt to calculate log(0) in the cost function, so all 1s are replaced with the largest
        # representable floating point value that is less than 1.
        return np.minimum(h, 0.9999999999999999)

    def _cost(self, coefs, X, y):
        log_loss = -np.mean(y*np.log(self._hypothesis(X, coefs)) + (1 - y)*np.log(1 - self._hypothesis(X, coefs)))
        regularization = self._regularization_param / (2 * X.shape[0]) * (coefs[1:]**2).sum()
        return log_loss + regularization
