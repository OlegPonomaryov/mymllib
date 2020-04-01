from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @staticmethod
    def _transform_to_numpy(X, y=None):
        # Though, for instance, Pandas DataFrame and Series can be used as NumPy arrays, doing this results in severe
        # decrease of performance, so features and target should be converted to pure NumPy arrays.
        X = np.asarray(X)
        if y is None:
            return X
        else:
            y = np.asarray(y)
            return X, y


class BaseSupervisedModel(BaseModel):
    """Base class for supervised models."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        pass
