"""Different activation functions for neural netwprks."""
from abc import ABC, abstractmethod
from mymllib.math.functions import sigmoid


class BaseActivationFunction(ABC):
    """Base class for activation functions."""

    @staticmethod
    @abstractmethod
    def activations(previous_activations, weights):
        """Calculate activations of the next layer.

        :param previous_activations: Activations of the previous layer
        :param weights: Matrix of weights
        :return: Activations of the next layer
        """
        pass


class SigmoidActivationFunction(BaseActivationFunction):
    """Base class for activation functions."""

    @staticmethod
    def activations(previous_activations, weights):
        """Calculate activations of the next layer.

        :param previous_activations: Activations of the previous layer
        :param weights: Matrix of weights
        :return: Activations of the next layer
        """
        z = previous_activations @ weights.T
        return sigmoid(z)

