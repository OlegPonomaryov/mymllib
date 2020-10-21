"""Different activation functions for neural networks."""
from abc import ABC, abstractmethod
from mymllib.math.functions import sigmoid, log_loss, softmax, softmax_loss


class BaseOutputActivation(ABC):
    """Base class for activation functions for output layer."""

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
        """Calculate loss function value. Neural network implementation expects its derivative with respect to the
           output activation function's argument to be (y_pred - y_actual).

        :param a: Activation values
        :param y: Actual values
        :return: Loss value
        """
        pass


class SigmoidOutput(BaseOutputActivation):
    """Sigmoid activation function for output layer."""

    @staticmethod
    def activations(x):
        """Calculate sigmoid activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return sigmoid(x)

    @staticmethod
    def loss(a, y):
        """Calculate binary cross entropy loss function value.

        :param a: Activation values
        :param y: Actual values
        :return: Loss value
        """
        return log_loss(a, y)


class SoftmaxOutput(BaseOutputActivation):
    """Softmax activation function for output layer."""

    @staticmethod
    def activations(x):
        """Calculate softmax activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return softmax(x)

    @staticmethod
    def loss(a, y):
        """Calculate cross entropy loss function value.

        :param a: Activation values
        :param y: Actual values
        :return: Loss value
        """
        return softmax_loss(a, y)


class IdentityOutput(BaseOutputActivation):
    """Identity activation function for output layer."""

    @staticmethod
    def activations(x):
        """Calculate identity activation function value.

        :param x: Function argument
        :return: Activation value
        """
        return x

    @staticmethod
    def loss(a, y):
        """Calculate squared error loss function value.

        :param a: Activation values
        :param y: Actual values
        :return: Loss value
        """
        return 0.5*(a - y)**2
