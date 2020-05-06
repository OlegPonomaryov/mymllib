import numpy as np
from scipy.spatial.distance import cdist
from mymllib import BaseUnsupervisedModel


class KMeans(BaseUnsupervisedModel):
    """Implementation of k-means clustering.

    :param runs_count: How many runs of the algorithm to perform
    :param max_iter: Maximum number of iterations per single algorithm run
    :param accuracy: Minimum relative cost decrease required to stop the algorithm before reaching maximum iterations
        count
    """

    def __init__(self, clusters_count, runs_count=100, max_iter=300, accuracy=1E-5):
        self._clusters_count = clusters_count
        self._runs_count = runs_count
        self._max_iter = max_iter
        self._accuracy = accuracy
        self.cluster_centroids = None

    def fit(self, X):
        """Train the model.

        :param X: Features values of data examples
        """
        X = self._check_fit_data(X)
        min_cost = best_cluster_centroids = None
        for i in range(self._runs_count):
            cluster_centroids = self._single_run(X)
            distances = self._find_distances(X, cluster_centroids)
            cost = self._cost(distances)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_cluster_centroids = cluster_centroids
        self.cluster_centroids = best_cluster_centroids

    def predict(self, X):
        """Split the data into clusters.

        :param X: Features values of data examples
        :return: Indices of clusters to which each example was assigned
        """
        X = self._check_predict_data(X)
        distances = self._find_distances(X, self.cluster_centroids)
        return self._find_closest_centroids(distances)

    def _single_run(self, X):
        cluster_centroids = self._random_init(X)
        previous_cost = None
        for iteration in range(self._max_iter):
            distances = self._find_distances(X, cluster_centroids)

            cost = self._cost(distances)
            if self._check_cost_decrease(cost, previous_cost):
                break
            previous_cost = cost

            assigned_centroids = self._find_closest_centroids(distances)
            cluster_centroids = self._get_new_centroids(X, assigned_centroids, cluster_centroids)
        return cluster_centroids

    def _random_init(self, X):
        if self._clusters_count >= X.shape[0]:
            raise ValueError(f"Clusters count ({self._clusters_count}) is greater or equal to "
                             f"the number of training examples ({X.shape[0]})")
        return X[np.random.choice(X.shape[0], self._clusters_count, replace=False)]

    def _find_distances(self, X, cluster_centroids):
        return cdist(X, cluster_centroids, metric='sqeuclidean')

    def _check_cost_decrease(self, current_cost, previous_cost):
        if previous_cost is not None:
            if current_cost > previous_cost:
                # It is not possible for the cost function to sometimes increase. There must be a bug in the code.
                raise ValueError("Cost value increased compared to the previous iteration")
            elif (previous_cost - current_cost) / previous_cost <= self._accuracy:
                return True
        return False

    def _find_closest_centroids(self, distances):
        return np.argmin(distances, axis=1)

    def _get_new_centroids(self, X, assigned_centroids, old_cluster_centroids):
        new_cluster_centroids = list()
        for i in range(old_cluster_centroids.shape[0]):
            assigned_examples = X[assigned_centroids == i]
            # A cluster centroid will be added to the list of new centroids only if at least one data sample was
            # assigned to it
            if assigned_examples.shape[0] > 0:
                new_cluster_centroids.append(assigned_examples.mean(axis=0))
        return np.asarray(new_cluster_centroids)

    def _cost(self, distances):
        return np.min(distances, axis=1).mean()
