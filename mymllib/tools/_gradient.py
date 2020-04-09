import numpy as np


def gradient(x, func, args=(), h=1E-5):
    """Numerically calculates a gradient using symmetric difference quotient.

    :param x: N-dimensional array at which to calculate the gradient
    :param func: A function of which to calculate the gradient
    :param args: Other arguments of the function
    :param h: A change in x used to calculate the gradient
    :return: Approximate gradient of the function
    """
    gradient = np.empty_like(x)
    for index in np.ndindex(x.shape):
        a = x.copy()
        a[index] += h
        b = x.copy()
        b[index] -= h
        gradient[index] = (func(a, *args) - func(b, *args)) / (2 * h)
    return gradient
