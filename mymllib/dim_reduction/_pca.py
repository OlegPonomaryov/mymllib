from scipy.linalg import svd
from mymllib import Base


class PCA(Base):
    """Principal component analysis (PCA) implementation.

    :param components_count: Number of components to return (can be used only if min_retained_variance is None)
    :param min_retained_variance: Minimum retained variance used to automatically select number of components (can be
        used only if components count is None)
    """

    def __init__(self, components_count=None, min_retained_variance=None):
        if components_count is not None and min_retained_variance is not None:
            raise ValueError("Setting both components count and minimum retained variance is not allowed")

        if min_retained_variance is not None and min_retained_variance >= 1:
            raise ValueError("Minimum retained variance should be less than 1")

        self.components_count = components_count
        self._min_retained_variance = min_retained_variance

        self.singular_vectors = None
        self.singular_values = None
        self.eigenvalues = None
        self.retained_variance = None

    def fit(self, X):
        """Calculate singular vectors that will be used for dimensionality reduction.

        :param X: A data to use for calculating singular vectors
        :return: An instance of the PCA class for which fit() was called
        """
        X = self._check_fit_data(X)
        u, s, v = svd((X.T@X)/X.shape[0])
        relative_variance = s / s.sum()

        if self.components_count is not None:
            components_count = self.components_count
        else:
            retained_variance = 0
            for i in range(relative_variance.size):
                if retained_variance >= self._min_retained_variance:
                    components_count = i
                    break
                retained_variance += relative_variance[i]

        self.components_count = components_count
        self.singular_vectors = u[:, :components_count]
        self.singular_values = s[:components_count]
        self.eigenvalues = self.singular_values**2
        self.retained_variance = relative_variance[:components_count]
        return self

    def transform(self, X):
        """Reduce dimensions of data using previously calculated singular vectors.

        :param X: A data to transform
        :return: Transformed data
        """
        X = self._check_data(X)
        return X@self.singular_vectors
