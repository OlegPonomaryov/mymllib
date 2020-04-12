from ._functions import to_numpy


class DataScaler:
    """A class for scaling data by mean and standard deviation."""

    def __init__(self):
        self._mean = None
        self._standard_deviation = None

    def fit(self, X):
        """Calculate mean and standard deviation that will be used for scaling.

        :param X: A data to use for calculating mean and standard deviation
        :return: An instance of DataScaler for which fit() was called
        """
        X = to_numpy(X)
        self._mean = X.mean(axis=0)
        self._standard_deviation = X.std(axis=0)
        # Replace 0 standard deviations with 1 to avoid division by 0
        self._standard_deviation[self._standard_deviation == 0] = 1
        return self

    def scale(self, X):
        """Scale a data by previously calculated mean and standard deviation.

        :param X: A data to scale
        :return: Scaled data
        """
        X = to_numpy(X)
        if X.shape[1] != len(self._mean):
            raise ValueError("Data passed to transform() doesn't have same columns count as the one passed to fit()")
        return (X - self._mean) / self._standard_deviation
