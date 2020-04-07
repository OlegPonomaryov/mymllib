from numpy import identity
from numpy.linalg import inv, pinv
from ..preprocessing import add_intercept
from ._base_regression import BaseRegression


class LinearRegression(BaseRegression):
    """Linear regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function (if None than analytical method will be used)
    """

    def __init__(self, regularization_param=0, optimizer=None):
        super().__init__(regularization_param=regularization_param, optimizer=optimizer)
        self._optimizer = optimizer

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X, y = LinearRegression._transform_to_numpy(X, y)
        X = add_intercept(X)
        if self._optimizer is not None:
            super().fit(X, y)
        else:
            self._coefs = self._normal_equation(X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        X = LinearRegression._transform_to_numpy(X)
        return super().predict(add_intercept(X))

    def _hypothesis(self, X, coefs):
        return X @ coefs

    def _cost(self, coefs, X, y):
        return (((self._hypothesis(X, coefs) - y)**2).sum() +
                self._regularization_param*(coefs[1:]**2).sum()) / (2 * X.shape[0])

    def _normal_equation(self, X, y):
        if self._regularization_param > 0:
            regularization_matrix = self._regularization_param*identity(X.T.shape[0])
            regularization_matrix[0, 0] = 0
            # If the regularization parameter is greater than 0, a matrix in the equation is always invertible, so there
            # is no need to calculate a pseudo-inverse (pinv()) instead of an ordinary inverse (inv())
            return inv(X.T @ X + regularization_matrix) @ X.T @ y
        else:
            # X.T @ X is symmetric and if a matrix with only real entries is symmetric than it is Hermitian
            return pinv(X.T @ X, hermitian=True) @ X.T @ y
