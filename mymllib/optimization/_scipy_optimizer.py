from ._base_optimizer import BaseOptimizer
from scipy.optimize import minimize


class SciPyOptimizer(BaseOptimizer):
    """An adapter to use optimizers from the SciPy library.

    :param method: A method name to pass into scipy.optimize.minimize
    :param options: Options to pass into scipy.optimize.minimize
    """
    def __init__(self, method, options=None):
        self._method = method
        self._options = options

    def minimize(self, func, grad, x0, args=()):
        """Find an optimal parameters array to minimize a function.

        :param func: A function to minimize
        :param grad: A gradient of the function
        :param x0: Initial value of the parameters array to start from
        :param args: Other arguments of the function
        :return: The best arguments array optimizer was able to find to minimize the function
        """
        return minimize(func, x0, args=args, method=self._method, jac=grad, options=self._options).x
