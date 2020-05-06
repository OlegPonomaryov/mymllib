"""Tests for the KMeans class."""
import pytest
import numpy as np
from mymllib.clustering import KMeans
from mymllib.preprocessing import to_numpy


X = [[5, 0],
     [4, 1],
     [6, 2],

     [0, 6],
     [1, 5],
     [2, 7],

     [6, 6],
     [7, 5],
     [4, 7]]
actual_clusters_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("X", [to_numpy(X)])
@pytest.mark.parametrize("clusters_count", [len(X), len(X) + 1])
def test_random_init__not_enough_samples(X, clusters_count):
    model = KMeans(clusters_count)

    with pytest.raises(ValueError):
        model._random_init(X)


@pytest.mark.parametrize("X", [to_numpy(X)])
@pytest.mark.parametrize("clusters_count", [1, 2, 3, len(X) - 1])
def test_random_init(X, clusters_count):
    model = KMeans(clusters_count)

    cluster_centroids = model._random_init(X)

    # Check that correct number of centroids was returned
    assert cluster_centroids.shape[0] == clusters_count

    # Check that all centroids are unique
    assert np.unique(cluster_centroids, axis=0).shape[0] == cluster_centroids.shape[0]

    # Check that all centroids were selected from the samples
    X_list = X.tolist()
    for centroid in cluster_centroids:
        assert centroid.tolist() in X_list


def test_fit_predict():
    model = KMeans(3)

    model.fit(X)
    predicted_clusters_labels = model.predict(X)

    X_np = to_numpy(X)
    actual_clusters = [X_np[actual_clusters_labels == label].tolist() for label in np.unique(actual_clusters_labels)]
    for label in np.unique(predicted_clusters_labels):
        predicted_cluster = X_np[predicted_clusters_labels == label].tolist()
        assert predicted_cluster in actual_clusters
