import numpy as np
from ..optimization import gradient_descent


class LinearRegression:
    """Linear regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regression is used)
    :param use_gradient_descent: Whether to use gradient descent instead of normal equation to fit the model
    :param learning_rate: Initial learning rate of gradient descent (used only if use_gradient_descent set to True, can be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent (used only if use_gradient_descent set to True)
    :param max_iterations: Maximum iterations count of gradient descent (used only if use_gradient_descent set to True)
    """

    def __init__(self, regularization_param=0, use_gradient_descent=False,
                 learning_rate=1, accuracy=1E-5, max_iterations=10000):
        assert regularization_param >= 0, "Regularization parameter must be >= 0, but was negative."
        self._coefs = None
        self._regularization_param = regularization_param
        self._use_gradient_descent = use_gradient_descent
        self._learning_rate = learning_rate
        self._accuracy = accuracy
        self._max_iterations = max_iterations

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X = LinearRegression._add_intercept(X)

        if self._use_gradient_descent:
            self._coefs = gradient_descent(self._cost, self._cost_gradient, np.zeros(X.shape[1]), (X, y),
                                           self._learning_rate, self._accuracy, self._max_iterations)
        else:
            self._coefs = self._normal_equation(X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        return LinearRegression._linear_func(LinearRegression._add_intercept(X), self._coefs)

    def _cost(self, coefs, X, y):
        return (((LinearRegression._linear_func(X, coefs) - y)**2).sum() +
                self._regularization_param*(coefs[1:]**2).sum()) / X.shape[0]

    def _cost_gradient(self, coefs, X, y):
        # Intercept should not be regularized, so it is set to 0 in a copy of the coefficients vector
        coefs_without_intercept = coefs.copy()
        coefs_without_intercept[0] = 0
        return ((LinearRegression._linear_func(X, coefs) - y)@X +
                self._regularization_param * coefs_without_intercept) / X.shape[0]

    def _normal_equation(self, X, y):
        features_count = X.T.shape[0]
        regularization_matrix = self._regularization_param*np.identity(features_count)
        regularization_matrix[0, 0] = 0
        return np.linalg.pinv(X.T @ X + regularization_matrix) @ X.T @ y

    @staticmethod
    def _linear_func(X, coefs):
        return X@coefs

    @staticmethod
    def _add_intercept(X):
        """Add intercept feature (all-ones)

        :param X: Features
        :return: Features with intercept feature
        """
        samples_count = X.shape[0]
        intercept_column = np.ones((samples_count, 1))
        return np.hstack((intercept_column, X))
