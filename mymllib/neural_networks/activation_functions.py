"""Different activation functions for neural netwprks."""
from abc import ABC, abstractmethod
from mymllib.math.functions import sigmoid


class BaseActivationFunction(ABC):
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


class SigmoidActivationFunction(BaseActivationFunction):
    """Base class for activation functions."""

    @staticmethod
    def activations(x):
        """Calculate activation function value."""
        return sigmoid(x)

    @staticmethod
    def derivative(x):
        """Calculate activation function derivative."""
        a = SigmoidActivationFunction.activations(x)
        return a * (1 - a)
