"""Various mathematical tools."""
import numpy as np


def gradient(x, func, args=(), h=1E-5):
    """Numerically calculates a gradient using symmetric difference quotient.

    :param x: N-dimensional array at which to calculate the gradient
    :param func: A function of which to calculate the gradient
    :param args: Other arguments of the function
    :param h: A change in x used to calculate the gradient
    :return: Approximate gradient of the function
    """
    # It is necessary for gradient, a and b to be of float type, because if they inherit int type from x, assigning
    # float values to their elements will result in 'truncating' this values (e.g. after running a[0] = 4.9999 value of
    # a[0] will be equal to 4)
    gradient = np.empty_like(x, dtype=float)
    for index in np.ndindex(x.shape):
        a = x.astype(float)
        b = x.astype(float)
        a[index] += h
        b[index] -= h
        gradient[index] = (func(a, *args) - func(b, *args)) / (2 * h)
    return gradient
