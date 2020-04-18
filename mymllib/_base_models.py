from abc import ABC, abstractmethod
import numpy as np
from mymllib.preprocessing import to_numpy
from mymllib.optimization import LBFGSB


class BaseModel(ABC):
    pass


class BaseSupervisedModel(BaseModel):
    """Base class for supervised models.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function
    """

    def __init__(self, regularization_param, optimizer=LBFGSB()):
        assert regularization_param >= 0, "Regularization parameter must be >= 0, but was negative."
        self._regularization_param = regularization_param
        self._params = None
        self._optimizer = optimizer

    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        self._params = self._optimize_params(X, y, np.zeros(X.shape[1]))

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        if X.shape[1] != self._params.shape[0]:
            raise ValueError(f"Expected {self._params.shape[0]} features, but {X.shape[1]} received")

        return self._hypothesis(X, self._params)

    def _optimize_params(self, X, y, initial_params):
        return self._optimizer.minimize(self._cost, self._cost_gradient, initial_params, (X, y))

    @staticmethod
    def _check_data(X, y):
        """Check that X and y have correct dimensionality and matching shapes and convert them to NumPy arrays.

        :param X: Features values
        :param y: Target values
        :return: X and y as NumPy arrays.
        """
        X, y = to_numpy(X, y)

        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        if y.ndim != 1:
            raise ValueError("Target values (y) should be a one-dimensional array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Samples count in features values (X) and target values (y) don't match each other")

        return X, y

    @abstractmethod
    def _hypothesis(self, X, params):
        pass

    @abstractmethod
    def _cost(self, params, X, y):
        pass

    @abstractmethod
    def _cost_gradient(self, params, X, y):
        pass
