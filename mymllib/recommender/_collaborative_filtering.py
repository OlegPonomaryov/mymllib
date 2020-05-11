import numpy as np
from mymllib.optimization import LBFGSB, unroll, undo_unroll
from mymllib.tools import glorot_init
from mymllib.preprocessing import to_numpy


class CollaborativeFiltering:
    """Collaborative filtering algorithm implementation.

    :param features_count: Number of item features to learn
    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function (if None than analytical method will be used)
    """
    def __init__(self, features_count, regularization_param=0, optimizer=LBFGSB()):
        self._features_count = features_count
        self._regularization_param = regularization_param
        self._optimizer = optimizer
        self._users_count = None
        self._items_count = None
        self._users_params = None
        self._items_params = None

    def fit(self, Y):
        """Train the model.

        :param Y: Ratings matrix with columns corresponding to users and rows corresponding to items
        """
        Y = self._check_fit_data(Y)
        self._items_count, self._users_count = Y.shape
        initial_params = unroll(glorot_init(self._get_params_shapes()))
        params = self._optimizer.minimize(self._cost, self._cost_gradient, initial_params, (Y,))
        self._users_params, self._items_params = self._undo_params_unroll(params)

    def predict(self, X):
        """Predict ratings.

        :param X: A matrix with users indices as the first column and items indices as the second one so that each row
            forms a user-item pair for which rating should be predicted
        :return: Predicted ratings
        """
        X = self._check_predict_data(X)
        Y = self._hypothesis(self._users_params, self._items_params)
        return Y[X[:, 1], X[:, 0]]

    def _check_fit_data(self, Y):
        """Check that Y has correct dimensionality and convert it to a NumPy array.

        :param Y: Features values
        :return: Y as a NumPy array
        """
        Y = to_numpy(Y)

        if Y.ndim != 2:
            raise ValueError("Features values (Y) should be a two-dimensional array")

        return Y

    def _check_predict_data(self, X):
        """Check that X has correct shape and convert it to a NumPy array.

        :param X: Features values
        :return: X as a NumPy array
        """
        X = to_numpy(X)

        if X.ndim != 2:
            raise ValueError("Features values (X) should be a two-dimensional array")

        if X.shape[1] != 2:
            raise ValueError(f"X should contain 2 columns")

        return X

    def _undo_params_unroll(self, params):
        return undo_unroll(params, self._get_params_shapes())

    def _get_params_shapes(self):
        return (self._users_count, self._features_count), (self._items_count, self._features_count)

    def _hypothesis(self, users_params, items_params):
        return items_params @ users_params.T

    def _cost(self, params, Y):
        users_params, items_params = self._undo_params_unroll(params)
        return (np.sum(self._error(self._hypothesis(users_params, items_params), Y)**2) +
                self._regularization_param * np.sum(params**2)) / 2

    def _cost_gradient(self, params, Y):
        users_params, items_params = self._undo_params_unroll(params)
        error = self._error(self._hypothesis(users_params, items_params), Y)
        users_params_grad = (items_params.T @ error).T + self._regularization_param * users_params
        items_params_grad = (users_params.T @ error.T).T + self._regularization_param * items_params
        return unroll((users_params_grad, items_params_grad))

    def _error(self, predicted, actual):
        error = predicted - actual
        error[np.isnan(actual)] = 0
        return error
