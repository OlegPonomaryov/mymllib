import numpy as np
from ..preprocessing import add_intercept
from ..regression._linear_regression import BaseRegression


class LogisticRegression:
    """Logistic regression implementation.

    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is used)
    :param learning_rate: Initial learning rate of gradient descent (can be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent
    :param max_iterations: Maximum iterations count of gradient descent
    """

    def __init__(self, regularization_param=0, learning_rate=1, accuracy=1E-5, max_iterations=10000):
        self._regularization_param = regularization_param
        self._learning_rate = learning_rate
        self._accuracy = accuracy
        self._max_iterations = max_iterations

        self._labels = None
        self._binary_classifiers = None
        self._labels_dtype = None

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        labels = y.unique()
        if len(labels) < 2:
            raise ValueError("There should be at least 2 different classes")

        self._labels = labels
        self._binary_classifiers = dict()
        self._labels_dtype = y.dtype

        # If there are only two classes it is effectively a binary classification and only one classifier is needed
        if len(self._labels) == 2:
            labels = labels[:1]

        for label in labels:
            self._binary_classifiers[label] = self._fit_binary_classifier(X, y, label)

    def predict(self, X):
        one_vs_all = dict()
        for label in self._binary_classifiers:
            one_vs_all[label] = self._binary_classifiers[label].predict(X)

        # In binary classification probabilities of the second class are just inverses of probabilities of the first one
        if len(self._labels) == 2:
            one_vs_all[self._labels[1]] = np.ones(X.shape[0]) - one_vs_all[self._labels[0]]

        predictions = np.ndarray(X.shape[0], dtype=self._labels_dtype)
        for i in range(len(predictions)):
            max_probability = -1
            for label in one_vs_all:
                if one_vs_all[label][i] > max_probability:
                    max_probability = one_vs_all[label][i]
                    predictions[i] = label
        return predictions

    def _fit_binary_classifier(self, X, y, label):
        binary_y = (y == label) * 1
        binary_classifier = _BinaryLogisticRegression(
            threshold=None, regularization_param=self._regularization_param, learning_rate=self._learning_rate,
            accuracy=self._accuracy, max_iterations=self._max_iterations)
        binary_classifier.fit(X, binary_y)
        return binary_classifier


class _BinaryLogisticRegression(BaseRegression):
    """Simple binary logistic regression implementation.

    :param threshold: A threshold for classification (if None than raw probabilities are returned)
    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is used)
    :param learning_rate: Initial learning rate of gradient descent (can be automatically reduced if too high)
    :param accuracy: Accuracy of gradient descent
    :param max_iterations: Maximum iterations count of gradient descent
    """

    def __init__(self, threshold=0.5, regularization_param=0, learning_rate=1, accuracy=1E-5, max_iterations=10000):
        super().__init__(regularization_param, learning_rate, accuracy, max_iterations)
        self._threshold = threshold

    def fit(self, X, y):
        """Train the model.

        :param X: Features
        :param y: Target values
        """
        X = add_intercept(X)
        super().fit(X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features
        :return: Target values
        """
        prediction = super().predict(add_intercept(X))
        return (prediction >= self._threshold) * 1 if self._threshold is not None else prediction

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
        log_loss = -np.mean(y*np.log(self._hypothesis(X, coefs)) + (1 - y)*np.log(1 - self._hypothesis(X, coefs)))
        regularization = self._regularization_param / (2 * X.shape[0]) * (coefs[1:]**2).sum()
        return log_loss + regularization
