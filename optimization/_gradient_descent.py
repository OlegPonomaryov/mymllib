import numpy as np


def gradient_descent(func, func_grad, x0, args, learning_rate=1, accuracy=1E-5, max_iterations=10000):
    """Find optimal value of arguments to minimize a function.

    :param func: A function to minimize
    :param func_grad: A gradient of the function
    :param x0: Initial value of arguments to start from
    :param args: Other arguments of the function
    :param learning_rate: Initial learning rate of gradient descent (will be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent
    :param max_iterations: Maximum iterations count of gradient descent
    :return: Best found values of arguments to minimize the function
    """
    x = x0
    value = func(x, *args)

    for iteration in range(max_iterations):
        previous_x = x
        previous_value = value

        gradient = func_grad(x, *args)

        # If absolute values of all partial derivatives are small enough, than coefficients are close enough to the
        # minimum and there is no need to continue gradient descent
        if np.sum(np.abs(gradient) > accuracy) == 0:
            break

        x = x - learning_rate * gradient

        # If new x made the function value higher, it is reverted to previous x and learning rate is decreased
        value = func(x, *args)
        if value > previous_value:
            x = previous_x
            value = previous_value
            learning_rate /= 2

    return x
