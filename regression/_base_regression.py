import numpy as np
from ..optimization import gradient_descent


class BaseRegression:
    """Base class for regressions that uses gradient descent to fit the model.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is used)
    :param learning_rate: Initial learning rate of gradient descent (can be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent
    :param max_iterations: Maximum iterations count of gradient descent
    """

    def __init__(self, regularization_param, learning_rate=1, accuracy=1E-5, max_iterations=10000):
        assert regularization_param >= 0, "Regularization parameter must be >= 0, but was negative."
        self._regularization_param = regularization_param
        self._coefs = None
        self._learning_rate = learning_rate
        self._accuracy = accuracy
        self._max_iterations = max_iterations

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        self._coefs = gradient_descent(self._cost, self._cost_gradient,
                                       np.zeros((X.shape[1], y.shape[1]) if y.ndim >= 2 else X.shape[1]),
                                       (X, y),
                                       self._learning_rate, self._accuracy, self._max_iterations)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        return self._hypothesis(X, self._coefs)

    def _hypothesis(self, X, coefs):
        pass

    def _cost(self, coefs, X, y):
        pass

    def _cost_gradient(self, coefs, X, y):
        # Intercept should not be regularized, so it is set to 0 in a copy of the coefficients vector
        coefs_without_intercept = coefs.copy()
        coefs_without_intercept[0] = 0
        return (X.T@(self._hypothesis(X, coefs) - y) +
                self._regularization_param * coefs_without_intercept) / X.shape[0]
