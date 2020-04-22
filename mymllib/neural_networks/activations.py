"""Different activation functions for neural netwprks."""
from abc import ABC, abstractmethod
from mymllib.math.functions import sigmoid


class BaseActivation(ABC):
    """Base class for activation functions."""

    @staticmethod
    @abstractmethod
    def activations(x):
        """Calculate activation function value."""
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        """Calculate activation function derivative."""
        pass


class Sigmoid(BaseActivation):
    """Base class for activation functions."""

    @staticmethod
    def activations(x):
        """Calculate activation function value."""
        return sigmoid(x)

    @staticmethod
    def derivative(x):
        """Calculate activation function derivative."""
        a = Sigmoid.activations(x)
        return a * (1 - a)
