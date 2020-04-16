from ._base_optimizer import BaseOptimizer
from scipy.optimize import minimize


class LBFGSB(BaseOptimizer):
    """An optimizer-like wrapper for SciPy L-BFGS-B implementation.

    :param options: Options to pass into scipy.optimize.minimize
    """
    def __init__(self, options=None):
        self._options = options

    def minimize(self, func, grad, x0, args=()):
        """Find an optimal arguments array to minimize a function.

        :param func: A function to minimize
        :param grad: A gradient of the function
        :param x0: Initial value of an arguments array to start from
        :param args: Other arguments of the function
        :return: The best arguments array optimizer was able to find to minimize the function
        """
        return minimize(func, x0, args=args, method='L-BFGS-B', jac=grad, options=self._options).x
