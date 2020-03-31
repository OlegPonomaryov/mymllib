import numpy as np
from ..optimization import GradientDescent
from ..preprocessing import add_intercept
from ..regression._linear_regression import BaseRegression


class LogisticRegressionWithMatrix(BaseRegression):
    """Logistic regression implementation that uses coefficients matrix for multiclass problems instead of separate binary classificators.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is used)
    :param optimizer: An optimizer to use for minimizing a cost function
    :param predict_probabilities: Whether to return probabilities of samples to belong to each of the classes (ordered by their labels) instead of chosen classes
    """

    def __init__(self, regularization_param=0, optimizer=GradientDescent(), predict_probabilities=False):
        super().__init__(regularization_param=regularization_param, optimizer=optimizer)
        self._predict_probabilities = predict_probabilities
        self._labels = None

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X, y = LogisticRegressionWithMatrix._transform_to_numpy(X, y)

        self._labels, Y = LogisticRegressionWithMatrix._one_hot(y)

        X = add_intercept(X)
        super().fit(X, Y)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        X = LogisticRegressionWithMatrix._transform_to_numpy(X)

        predictions = super().predict(add_intercept(X))
        if self._predict_probabilities:
            return predictions
        else:
            if len(self._labels) == 2:
                predictions = (predictions >= 0.5) * 1
            else:
                predictions = np.argmax(predictions, axis=1)
            return self._labels.take(predictions)

    def _hypothesis(self, X, coefs):
        z = X @ coefs

        # A value of exp(n) exceeds a capacity of double-precision floating-point variables if n is higher than
        # approximately 709.7. For np.exp() this results in warning message and inf return value, which also makes the
        # hypothesis to return 0 resulting in an attempt to calculate log(0) in the cost function. To avoid this, all
        # values from z that are lower than -709.7 (because z is used with '-' in np.exp()) are replaced with -709.7.
        z = np.maximum(z, -709.7)

        h = 1 / (1 + np.exp(-z))

        # Values that are very close to 1 (like 0.9999999999999999999999) cannot be stored in double-precision floating-
        # point variables due to their significant digits limitation and are rounded to 1. But returning exactly 1 will
        # result in an attempt to calculate log(0) in the cost function, so all 1s are replaced with the largest
        # representable floating point value that is less than 1.
        return np.minimum(h, 0.9999999999999999)

    def _cost(self, coefs, X, y):
        coefs = LogisticRegressionWithMatrix._reshape_coefs(coefs, X, y)

        log_loss = y*np.log(self._hypothesis(X, coefs)) + (1 - y)*np.log(1 - self._hypothesis(X, coefs))
        # np.mean() is not used for log_loss because for multiclass problems it will divide the sum not only by number
        # of samples, but also by number of classes and in this case the cost function won't match its gradient defined
        # in the BaseRegression class.
        log_loss = -np.sum(log_loss) / X.shape[0]

        regularization = self._regularization_param / (2 * X.shape[0]) * (coefs[1:]**2).sum()
        return log_loss + regularization

    def _cost_gradient(self, coefs, X, y):
        original_coefs_ndim = coefs.ndim
        coefs = LogisticRegressionWithMatrix._reshape_coefs(coefs, X, y)

        gradient = super()._cost_gradient(coefs, X, y)

        # Return gradient flattened if coefficients were flattened
        return gradient.flatten() if original_coefs_ndim < coefs.ndim else gradient

    @staticmethod
    def _reshape_coefs(coefs, X, y):
        # For multiclass problems this implementation of linear regression uses a matrix of coefficients. However, some
        # optimization algorithms (like scipy.optimize.minimize used by optimization.LBFGSB) are designed to work only
        # with 1-dimensional arrays. So this function checks dimensionality of coefficients, X and y, determines whether
        # coefficients were flattened by an optimization algorithm and reshapes them back to the original matrix.
        return coefs if coefs.ndim > 1 or y.ndim == 1 else coefs.reshape((X.shape[1], y.shape[1]))

    @staticmethod
    def _one_hot(y):
        all_labels = np.unique(y)

        if len(all_labels) < 2:
            raise ValueError("There should be at least 2 different classes")

        labels = all_labels[1:] if len(all_labels) == 2 else all_labels
        return all_labels, np.vstack(tuple((y == label)*1 for label in labels)).T

    @staticmethod
    def _transform_to_numpy(X, y=None):
        # Though, for instance, Pandas DataFrame and Series can be used as NumPy arrays, doing this results in severe
        # decrease of performance, so features and target should be converted to pure NumPy arrays.
        X = np.asarray(X)
        if y is None:
            return X
        else:
            y = np.asarray(y)
            return X, y
