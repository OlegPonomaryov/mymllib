"""Different activation functions for neural networks."""
from abc import ABC, abstractmethod
from mymllib.math.functions import sigmoid


class BaseActivation(ABC):
    """Base class for activation functions."""

    @staticmethod
    @abstractmethod
    def activations(x):
        """Calculate activation function value.

        :param x: Function argument
        :return: Activation value
        """
        pass

    @staticmethod
    @abstractmethod
    def derivative(a):
        """Calculate activation function derivative.

        :param a: Activation value
        :return: Activation derivative
        """
        pass


class Sigmoid(BaseActivation):
    """Sigmoid activation functions."""

    @staticmethod
    def activations(x):
        """Calculate activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return sigmoid(x)

    @staticmethod
    def derivative(a):
        """Calculate activation function derivative.

        :param a: Activation value
        :return: Activation derivative
        """
        return a * (1 - a)
