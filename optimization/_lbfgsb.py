from ._base_optimizer import BaseOptimizer
from scipy.optimize import minimize


class LBFGSB(BaseOptimizer):
    """An optimizer-like wrapper for SciPy L-BFGS-B implementation."""

    def minimize(self, func, grad, x0, args):
        """Find an optimal arguments array to minimize a function.

        :param func: A function to minimize
        :param grad: A gradient of the function
        :param x0: Initial value of an arguments array to start from
        :param args: Other arguments of the function
        :return: The best arguments array optimizer was able to find to minimize the function
        """
        # The arguments array may be a matrix (like coefficients in classification.LogisticRegressionWithMatrix), but
        # scipy.optimize.minimize is designed to work with 1-dimensional arrays, so x0 is flattened before proceeding.
        original_shape = None
        if x0.ndim > 1:
            original_shape = x0.shape
            x0 = x0.flatten()

        result = minimize(func, x0, args=args, method='L-BFGS-B', jac=grad)
        return result.x if original_shape is None else result.x.reshape(original_shape)
