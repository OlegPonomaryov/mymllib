from abc import ABC, abstractmethod


class BaseModel(ABC):
    pass


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
