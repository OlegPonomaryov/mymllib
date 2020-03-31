import numpy as np
from ._base_optimizer import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """An optimizer that uses gradient descent algorithm.

    :param learning_rate: Initial learning rate of gradient descent (will be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent
    :param max_iterations: Maximum iterations count of gradient descent
    """

    def __init__(self, learning_rate=1, accuracy=1E-5, max_iterations=10000):
        self._learning_rate = learning_rate
        self._accuracy = accuracy
        self._max_iterations = max_iterations

    def minimize(self, func, grad, x0, args):
        """Find an optimal arguments array to minimize a function.

        :param func: A function to minimize
        :param grad: A gradient of the function
        :param x0: Initial value of an arguments array to start from
        :param args: Other arguments of the function
        :return: The best arguments array optimizer was able to find to minimize the function
        """
        learning_rate = self._learning_rate
        x = x0
        value = func(x, *args)

        for iteration in range(self._max_iterations):
            previous_x = x
            previous_value = value

            gradient = grad(x, *args)

            # If absolute values of all partial derivatives are small enough, than coefficients are close enough to the
            # minimum and there is no need to continue gradient descent
            if np.sum(np.abs(gradient) > self._accuracy) == 0:
                break

            x = x - learning_rate * gradient

            # If the new x made the function value higher, it is reverted to previous x and learning rate is decreased
            value = func(x, *args)
            if value > previous_value:
                x = previous_x
                value = previous_value
                learning_rate /= 2

        return x
