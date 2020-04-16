import numpy as np
from mymllib.optimization import LBFGSB
from mymllib._base_models import BaseSupervisedModel
from abc import abstractmethod


class BaseRegression(BaseSupervisedModel):
    """Base class for regressions.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function
    """

    def __init__(self, regularization_param, optimizer=LBFGSB()):
        assert regularization_param >= 0, "Regularization parameter must be >= 0, but was negative."
        self._regularization_param = regularization_param
        self._coefs = None
        self._optimizer = optimizer

    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        self._coefs = self._optimize_coefs(X, y, np.zeros(X.shape[1]))

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        if X.shape[1] != self._coefs.shape[0]:
            raise ValueError(f"Expected {self._coefs.shape[0]} features, but {X.shape[1]} received")

        return self._hypothesis(X, self._coefs)

    def _optimize_coefs(self, X, y, initial_coefs):
        return self._optimizer.minimize(self._cost, self._cost_gradient, initial_coefs, (X, y))

    @abstractmethod
    def _hypothesis(self, X, coefs):
        pass

    @abstractmethod
    def _cost(self, coefs, X, y):
        pass

    def _cost_gradient(self, coefs, X, y):
        # Intercept should not be regularized, so it is set to 0 in a copy of the coefficients vector
        coefs_without_intercept = coefs.copy()
        coefs_without_intercept[0] = 0
        return (X.T@(self._hypothesis(X, coefs) - y) +
                self._regularization_param * coefs_without_intercept) / X.shape[0]
