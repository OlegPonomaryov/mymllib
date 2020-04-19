import numpy as np
from mymllib.optimization import LBFGSB, unroll, undo_unroll
from mymllib.preprocessing import to_numpy, add_intercept
from mymllib.regression._linear_regression import BaseRegression
from mymllib.math.functions import sigmoid


class LogisticRegression(BaseRegression):
    """Logistic regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function
    :param predict_probabilities: Whether to return probabilities of samples to belong to each of the classes (ordered
        by their labels) instead of chosen classes
    :param all_at_once: Whether to optimize all binary classification subproblems of a multiclass problem as a single
        optimization target
    """

    def __init__(self, regularization_param=0, optimizer=LBFGSB(), predict_probabilities=False,
                 all_at_once=False):
        super().__init__(regularization_param=regularization_param, optimizer=optimizer)
        self._predict_probabilities = predict_probabilities
        self._all_at_once = all_at_once
        self._labels = None

    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        X, y = LogisticRegression._check_data(X, y)
        X = add_intercept(X)
        self._labels, Y = LogisticRegression._one_hot(y)

        if self._all_at_once:
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
        X = to_numpy(X)

        predictions = super().predict(add_intercept(X))
        if self._predict_probabilities:
            return predictions.flatten() if len(self._labels) == 2 else predictions
        else:
            if len(self._labels) == 2:
                predictions = (predictions >= 0.5) * 1
            else:
                predictions = np.argmax(predictions, axis=1)
            return self._labels.take(predictions).flatten()

    def _hypothesis(self, X, params):
        z = X @ params
        return sigmoid(z)

    def _cost(self, params, X, y):
        _, params = LogisticRegression._undo_params_unroll(params, X, y)

        log_loss = y*np.log(self._hypothesis(X, params)) + (1 - y)*np.log(1 - self._hypothesis(X, params))
        # np.mean() is not used for log_loss because for multiclass problems it will divide the sum not only by number
        # of samples, but also by number of classes and in this case the cost function won't match its gradient defined
        # in the BaseRegression class.
        log_loss = -np.sum(log_loss) / X.shape[0]

        regularization = self._regularization_param / (2 * X.shape[0]) * (params[1:]**2).sum()
        return log_loss + regularization

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

    @staticmethod
    def _one_hot(y):
        all_labels = np.unique(y)

        if len(all_labels) < 2:
            raise ValueError("There should be at least 2 different classes")

        labels = all_labels[1:] if len(all_labels) == 2 else all_labels
        return all_labels, np.vstack(tuple((y == label)*1 for label in labels)).T
