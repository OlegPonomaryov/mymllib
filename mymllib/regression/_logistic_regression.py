import numpy as np
from mymllib.optimization import LBFGSB, unroll, undo_unroll
from mymllib.preprocessing import add_intercept, one_hot
from mymllib.regression._linear_regression import BaseRegression
from mymllib.math.functions import sigmoid, log_cost, softmax, softmax_cost


class LogisticRegression(BaseRegression):
    """Logistic regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param use_softmax: Whether to use softmax function for multiclass problemы or to use logistic function for each of
        the classes (one vs all) and select one class that maximizes logistic function value
    :param optimizer: An optimizer to use for minimizing a cost function
    """

    def __init__(self, regularization_param=0, use_softmax=False, optimizer=LBFGSB()):
        super().__init__(regularization_param=regularization_param, optimizer=optimizer)
        self._labels = None
        self._use_softmax = use_softmax

    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        X, y = self._check_fit_data(X, y)
        X = add_intercept(X)
        self._labels, Y = one_hot(y)
        if len(self._labels) > 2 and self._use_softmax:
            initial_params = np.zeros((X.shape[1], Y.shape[1]))
            params = self._optimize_params(X, Y, unroll((initial_params,)))
            self._params = undo_unroll(params, (initial_params.shape,))[0]
        else:
            self._params = np.apply_along_axis(lambda y_bin: self._optimize_params(X, y_bin, np.zeros(X.shape[1])), 0, Y)

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        X = self._check_data(X)

        predictions = super().predict(add_intercept(X))
        if len(self._labels) == 2:
            predictions = (predictions >= 0.5) * 1
        else:
            predictions = np.argmax(predictions, axis=1)
        return self._labels.take(predictions).flatten()

    def _hypothesis(self, X, params):
        z = X @ params
        return softmax(z) if params.ndim == 2 and params.shape[1] > 1 else sigmoid(z)

    def _cost(self, params, X, y):
        _, params = LogisticRegression._undo_params_unroll(params, X, y)

        model_output = self._hypothesis(X, params)
        cost = softmax_cost(model_output, y) if params.ndim == 2 and params.shape[1] > 1 else log_cost(model_output, y)

        regularization = self._regularization_param / (2 * X.shape[0]) * (params[1:]**2).sum()
        return cost + regularization

    def _cost_gradient(self, params, X, y):
        params_were_unrolled, params = LogisticRegression._undo_params_unroll(params, X, y)
        gradient = super()._cost_gradient(params, X, y)
        # Return unrolled gradient if parameters were unrolled too
        return unroll((gradient,)) if params_were_unrolled else gradient

    @staticmethod
    def _undo_params_unroll(params, X, y):
        params_were_unrolled = y.ndim == 2 and params.ndim == 1
        params = undo_unroll(params, ((X.shape[1], y.shape[1]),))[0] if params_were_unrolled else params
        return params_were_unrolled, params
