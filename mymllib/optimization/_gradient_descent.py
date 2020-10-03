import numpy as np
from ._base_optimizer import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Gradient descent implementation with a built-in learning rate reduction mechanism.

    :param learning_rate: Initial learning rate of gradient descent (will be automatically reduced if too high)
    :param accuracy: The algorithm will stop if all elements of a gradient are less than or equal to the accuracy
    :param max_iterations: Maximum iterations count of gradient descent
    :param lr_reduce_patience: A n umber of iterations without the function value reduction required to reduce the
        learning rate (use None to turn learning rate reduction off)
    :param lr_reduce_factor: A factor by which to reduce the learning rate
    """

    def __init__(self, learning_rate=0.01, accuracy=1E-5, max_iterations=100,
                 lr_reduce_patience=3, lr_reduce_factor=0.1):
        if lr_reduce_patience is not None and lr_reduce_patience <= 0:
            raise ValueError(f"Learning rate reduction patience expected to be either None or greater than 0, "
                             f"but {lr_reduce_patience} received")
        if lr_reduce_factor >= 1:
            raise ValueError(f"Learning rate reduction factor expected to be less than 1, "
                             f"but {lr_reduce_factor} received")

        self._learning_rate = learning_rate
        self._accuracy = accuracy
        self._max_iterations = max_iterations
        self._lr_reduce_patience = lr_reduce_patience
        self._lr_reduce_factor = lr_reduce_factor

    def minimize(self, func, grad, x0, args=()):
        """Find an optimal parameters array to minimize a function.

        :param func: A function to minimize
        :param grad: A gradient of the function
        :param x0: Initial value of the parameters array to start from
        :param args: Other arguments of the function
        :return: The best arguments array optimizer was able to find to minimize the function
        """
        learning_rate = self._learning_rate
        best_x = x = x0
        best_value = func(x, *args)
        iters_without_improve = 0

        for iteration in range(self._max_iterations):
            gradient = grad(x, *args)

            # If absolute values of all partial derivatives are equal to 0 with specified accuracy, then parameters are
            # close enough to the minimum and there is no need to continue gradient descent.
            if np.abs(gradient).max() <= self._accuracy:
                break

            x = x - learning_rate * gradient

            # If new values of x haven't lead to decrease of the function value for the specified number of iteration,
            # the x is reverted to its previous best value and the learning rate is reduced
            value = func(x, *args)
            if value > best_value:
                iters_without_improve += 1
                if iters_without_improve >= self._lr_reduce_patience:
                    x = best_x
                    learning_rate *= self._lr_reduce_factor
            else:
                iters_without_improve = 0
                best_value = value
                best_x = x

        return best_x
