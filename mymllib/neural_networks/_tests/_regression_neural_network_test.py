"""Tests for the ClassificationNeuralNetwork class."""
import pytest
from numpy.testing import assert_allclose
from mymllib._test_data.regression import X_train, y_train, X_test, y_test
from mymllib.neural_networks import RegressionNeuralNetwork


@pytest.mark.parametrize("regularization_param", [0, 0.01])
def test_fit_predict(regularization_param):
    neural_network = RegressionNeuralNetwork(regularization_param=regularization_param)

    neural_network.fit(X_train, y_train)
    predictions = neural_network.predict(X_test)

    assert_allclose(predictions, y_test, rtol=1e-3)
