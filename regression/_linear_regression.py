import numpy as np


class LinearRegression:
    """Linear regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regression is used)
    :param use_gradient_descent: Whether to use gradient descent instead of normal equation to fit the model
    :param learning_rate: An initial learning rate of gradient descent (used only if use_gradient_descent set to True, can be automatically reduced if too high)
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
            self._gradient_descent(X, y)
        else:
            self._normal_equation(X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        return LinearRegression._linear_func(LinearRegression._add_intercept(X), self._coefs)

    def _gradient_descent(self, X, y):
        coefs = np.zeros(X.shape[1])
        cost = self._cost(X, y, coefs)
        rate = self._learning_rate

        for iteration in range(self._max_iterations):
            previous_coefs = coefs
            previous_cost = cost

            gradient = self._cost_gradient(X, y, coefs)

            # If absolute values of all partial derivatives are small enough, than coefficients are close enough to the
            # minimum and there is no need to continue gradient descent
            if np.sum(np.abs(gradient) > self._accuracy) == 0:
                break

            coefs = coefs - rate * gradient

            # If new coefficients made the cost higher, they are reverted to previous values and learning rate is
            # decreased
            cost = self._cost(X, y, coefs)
            if cost > previous_cost:
                coefs = previous_coefs
                cost = previous_cost
                rate /= 2

        self._coefs = coefs

    def _cost(self, X, y, coefs):
        return (((LinearRegression._linear_func(X, coefs) - y)**2).sum() +
                self._regularization_param*(coefs[1:]**2).sum()) / X.shape[0]

    def _cost_gradient(self, X, y, coefs):
        # Intercept should not be regularized, so it is set to 0 in a copy of the coefficients vector
        coefs_without_intercept = coefs.copy()
        coefs_without_intercept[0] = 0
        return ((LinearRegression._linear_func(X, coefs) - y)@X +
                self._regularization_param * coefs_without_intercept) / X.shape[0]

    def _normal_equation(self, X, y):
        features_count = X.T.shape[0]
        regularization_matrix = self._regularization_param*np.identity(features_count)
        regularization_matrix[0, 0] = 0
        self._coefs = np.linalg.pinv(X.T @ X + regularization_matrix) @ X.T @ y

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
