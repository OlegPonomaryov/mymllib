"""Tests for the LogisticRegression class."""
from mymllib.neural_networks import ClassificationNeuralNetwork
from mymllib._test_data.classification import X, y, y_text, y_bin, test_set_start
import pytest
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("y", [y_bin, y, y_text])
@pytest.mark.parametrize("hidden_layers", [(), (5,)])
def test_fit_predict(y, hidden_layers):
    neural_network = ClassificationNeuralNetwork(
        hidden_layers=hidden_layers, regularization_param=1)

    neural_network.fit(X[:test_set_start], y[:test_set_start])
    predictions = neural_network.predict(X)

    assert_array_equal(predictions, y)
