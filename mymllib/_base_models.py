from abc import ABC, abstractmethod
from mymllib.preprocessing import to_numpy


class BaseModel(ABC):
    pass


class BaseSupervisedModel(BaseModel):
    """Base class for supervised models."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        pass

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
