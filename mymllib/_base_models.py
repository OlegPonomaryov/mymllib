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
        self._features_count = None

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
        return self._hypothesis(X, self._params)

    def _optimize_params(self, X, y, initial_params):
        return self._optimizer.minimize(self._cost, self._cost_gradient, initial_params, (X, y))

    def _check_fit_data(self, X, y):
        """Check that X and y have correct dimensionality and matching shapes and convert them to NumPy arrays.

        :param X: Features values
        :param y: Target values
        :return: X and y as NumPy arrays
        """
        X, y = to_numpy(X, y)

        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        if y.ndim != 1:
            raise ValueError("Target values (y) should be a one-dimensional array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Samples count in features values (X) and target values (y) don't match each other")

        self._features_count = X.shape[1]

        return X, y

    def _check_predict_data(self, X):
        """Check that X has correct dimensionality and shape and convert it to a NumPy array.

        :param X: Features values
        :return: X as a NumPy array
        """
        X = to_numpy(X)

        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        if X.shape[1] != self._features_count:
            raise ValueError(f"Expected {self._features_count} features, but {X.shape[1]} received")

        return X

    @abstractmethod
    def _hypothesis(self, X, params):
        pass

    @abstractmethod
    def _cost(self, params, X, y):
        pass

    @abstractmethod
    def _cost_gradient(self, params, X, y):
        pass


class BaseUnsupervisedModel(BaseModel):
    """Base class for unsupervised models."""

    def fit(self, X):
        """Train the model.

        :param X: Features values
        """
        pass

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Predicted values
        """
        pass

    def _check_fit_data(self, X):
        """Check that X has correct dimensionality and convert it to a NumPy array.

        :param X: Features values
        :return: X as a NumPy array
        """
        X = to_numpy(X)

        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        self._features_count = X.shape[1]

        return X

    def _check_predict_data(self, X):
        """Check that X has correct dimensionality and shape and convert it to a NumPy array.

        :param X: Features values
        :return: X as a NumPy array
        """
        X = to_numpy(X)

        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        if X.shape[1] != self._features_count:
            raise ValueError(f"Expected {self._features_count} features, but {X.shape[1]} received")

        return X
