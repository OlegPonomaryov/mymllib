from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Base class for optimizers."""

    @abstractmethod
    def minimize(self, func, grad, x0, args=()):
        """Find an optimal arguments array to minimize a function.

        :param func: A function to minimize
        :param grad: A gradient of the function
        :param x0: Initial value of an arguments array to start from
        :param args: Other arguments of the function
        :return: The best arguments array optimizer was able to find to minimize the function
        """
        pass
