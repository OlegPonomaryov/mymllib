import numpy as np
from scipy.spatial import KDTree
from mymllib import BaseSupervisedModel
from mymllib.optimization import SciPyOptimizer, unroll, undo_unroll
from mymllib.tools import glorot_init
from mymllib.preprocessing import to_numpy


class CollaborativeFiltering(BaseSupervisedModel):
    """Collaborative filtering algorithm implementation.

    :param features_count: Number of item features to learn
    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function (if None than analytical method will be used)
    """
    def __init__(self, features_count, regularization_param=0, optimizer=SciPyOptimizer("L-BFGS-B")):
        self._features_count = features_count
        self._regularization_param = regularization_param
        self._optimizer = optimizer
        self._Y_source = None
        self._users_count = None
        self._items_count = None
        self._users = None
        self._users_params = None
        self._items = None
        self._items_params = None
        self._items_tree = None

    def fit(self, X, y):
        """Train the model.

        :param X: A matrix in which each row is a unique user-item pair (users must come in the first column)
        :param y: Ratings that correspond to each user-item pair from X
        """
        X, y = self._check_X_y(X, y)
        self._Y_source, self._users, self._items = self._build_ratings_matrix(X, y)
        self._items_count, self._users_count = self._Y_source.shape
        initial_params = unroll(glorot_init(self._get_params_shapes()))
        params = self._optimizer.minimize(self._cost, self._cost_gradient, initial_params, (self._Y_source,))
        self._users_params, self._items_params = self._undo_params_unroll(params)
        self._items_tree = KDTree(self._items_params)

    def predict(self, X):
        """Predict ratings.

        :param X: A matrix with users indices as the first column and items indices as the second one so that each row
            forms a user-item pair for which rating should be predicted
        :return: Predicted ratings
        """
        X = self._check_X(X)
        Y = self._hypothesis(self._users_params, self._items_params)
        y = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            user, item = X[i, 0], X[i, 1]

            try:
                user_index = self._users.index(user)
            except ValueError:
                user_index = -1

            try:
                item_index = self._items.index(item)
            except ValueError:
                item_index = -1

            # If the item is new, its rating is 0 for all users, no matter new or not
            if item_index < 0:
                y[i] = 0
            # If the user is new, rating is the average of all known ratings for the item
            elif user_index < 0:
                item_ratings = self._Y_source[item_index]
                y[i] = item_ratings[~np.isnan(item_ratings)].mean()
            else:
                y[i] = Y[item_index, user_index]
        return y

    def find_similar_items(self, item, count):
        if count < 1:
            raise ValueError(f"Similar items count should be greater than 0, but {count} received")

        item_index = self._items.index(item)
        item_params = self._items_params[item_index]

        # Because KDTree.query() treats the requested item as a new item not from the initial data set, the closest
        # neighbor to the item will be the item itself, so the requested number of neighbors should be greater by 1 than
        # the requested number of similar items
        similar_items_indices = self._items_tree.query(item_params, count + 1)[1][:len(self._items)].tolist()
        try:
            # The requested item is removed from the list of similar items
            similar_items_indices.remove(item_index)
        except ValueError:
            # If the requested item wasn't in the list (if, for instance, there were other items with exactly the same
            # parameters), the list is truncated to match the requested number of similar items
            similar_items_indices = similar_items_indices[:-1]
        return tuple(self._items[index] for index in similar_items_indices)

    def _check_X_y(self, X, y):
        X = self._check_X(X)
        y = to_numpy(y)

        if y.ndim != 1:
            raise ValueError(f"Expected {(X.shape[0],)} shape of ratings, but {y.shape} received")

        return X, y

    def _check_X(self, X):
        X = to_numpy(X)

        if X.ndim != 2:
            raise ValueError("User-item pairs (X) should be a two-dimensional array")

        if X.shape[1] != 2:
            raise ValueError(f"Expected 2 columns in user-item pairs (X), but {X.shape[1]} received")

        return X

    def _build_ratings_matrix(self, X, y):
        users = np.unique(X[:, 0]).tolist()
        users_count = len(users)
        items = np.unique(X[:, 1]).tolist()
        items_count = len(items)
        ratings_matrix = np.empty((items_count, users_count))
        ratings_matrix.fill(np.nan)
        for i in range(X.shape[0]):
            user, item = X[i, 0], X[i, 1]
            ratings_matrix[items.index(item), users.index(user)] = y[i]
        return ratings_matrix, users, items

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
