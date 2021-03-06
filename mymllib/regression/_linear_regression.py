from numpy.linalg import solve, lstsq
from mymllib.preprocessing import add_intercept
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

        :param X: Features values
        :param y: Target values
        """
        X, y = self._check_fit_data(X, y)
        X = add_intercept(X)
        if self._optimizer is not None:
            super().fit(X, y)
        else:
            self._params = self._normal_equation(X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        X = self._check_data(X)
        return super().predict(add_intercept(X))

    def _hypothesis(self, X, params):
        return X @ params

    def _cost(self, params, X, y):
        return (((self._hypothesis(X, params) - y)**2).sum() +
                self._regularization_param*(params[1:]**2).sum()) / (2 * X.shape[0])

    def _normal_equation(self, X, y):
        if self._regularization_param > 0:
            A = X.T @ X
            A[1:].flat[1::A.shape[1] + 1] += self._regularization_param
            # If the regularization parameter is greater than 0, the A matrix is always invertible and therefore has
            # full rank, so it is possible to use numpy.linalg.solve() which is faster than numpy.linalg.lstsq()
            return solve(A, X.T @ y)
        else:
            return lstsq(X, y, rcond=None)[0]
