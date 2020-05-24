"""Different activation functions for neural networks."""
from abc import ABC, abstractmethod
from mymllib.math.functions import sigmoid, log_loss, softmax, softmax_loss


class BaseOutputActivation(ABC):
    """Base class for activation functions for output layers."""

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
    def loss(a, y):
        """Calculate loss function value. Its derivative with respect to the activation function's argument should be
            equal to (activations - y_actual).

        :param a: Activation values
        :param y: Actual values
        :return: Loss value
        """
        pass


class SigmoidOutput(BaseOutputActivation):
    """Sigmoid activation functions."""

    @staticmethod
    def activations(x):
        """Calculate activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return sigmoid(x)

    @staticmethod
    def loss(a, y):
        """Calculate loss function value. Its derivative with respect to the activation function's argument should be
            equal to (activations - y_actual).

        :param a: Activation values
        :param y: Actual values
        :return: Loss value
        """
        return log_loss(a, y)


class SoftmaxOutput(BaseOutputActivation):
    """Softmax activation functions."""

    @staticmethod
    def activations(x):
        """Calculate activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return softmax(x)

    @staticmethod
    def loss(a, y):
        """Calculate loss function value. Its derivative with respect to the activation function's argument should be
            equal to (activations - y_actual).

        :param a: Activation values
        :param y: Actual values
        :return: Loss value
        """
        return softmax_loss(a, y)
