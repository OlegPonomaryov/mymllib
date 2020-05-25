"""Different activation functions for neural networks."""
from abc import ABC, abstractmethod
import numpy as np
from mymllib.math.functions import sigmoid, tanh


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
    """Sigmoid activation function."""

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


class Tanh(BaseActivation):
    """Hyperbolic tangent (tanh) activation function."""

    @staticmethod
    def activations(x):
        """Calculate activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return tanh(x)

    @staticmethod
    def derivative(a):
        """Calculate activation function derivative.

        :param a: Activation value
        :return: Activation derivative
        """
        return 1 - a**2


class ReLU(BaseActivation):
    """ReLU activation function."""

    @staticmethod
    def activations(x):
        """Calculate activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return np.maximum(0, x)

    @staticmethod
    def derivative(a):
        """Calculate activation function derivative.

        :param a: Activation value
        :return: Activation derivative
        """
        d = np.ones_like(a)
        d[a <= 0] = 0
        return d


class LeakyReLU(BaseActivation):
    """Leaky ReLU activation function."""

    @staticmethod
    def activations(x):
        """Calculate activation function value.

        :param x: Function argument
        :return: Activation value
        """
        y = x.astype(float, copy=True)
        y[y < 0] *= 0.01
        return y

    @staticmethod
    def derivative(a):
        """Calculate activation function derivative.

        :param a: Activation value
        :return: Activation derivative
        """
        d = np.ones_like(a)
        d[a <= 0] = 0.01
        return d
