"""Tests for the BaseNeuralNetwork class."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from mymllib.neural_networks import BaseNeuralNetwork
from mymllib.neural_networks.activation_functions import SigmoidActivationFunction


@pytest.mark.parametrize("input_layer, output_layer, hidden_layers", [
    (-16, 9, (27,)),
    (0, 9, (27,)),
    (16, -9, (27,)),
    (16, 0, (27,)),
    (16, 9, (-27,)),
    (16, 9, (0,))
])
def test_init_weights__zero_or_negative_layer_size(input_layer, output_layer, hidden_layers):
    with pytest.raises(ValueError):
        BaseNeuralNetwork._init_weights(input_layer, output_layer, hidden_layers=hidden_layers)


@pytest.mark.parametrize("input_layer, output_layer, hidden_layers, expected_shapes", [
    (5, 3, (), ((3, 6),)),
    (16, 9, (27, 27), ((27, 17), (27, 28), (9, 28)))
])
def test_init_weights(input_layer, output_layer, hidden_layers, expected_shapes):
    result = BaseNeuralNetwork._init_weights(input_layer, output_layer, hidden_layers=hidden_layers)

    for i in range(len(expected_shapes)):
        # Verify that weights matrix has correct shape
        assert_array_equal(result[i].shape, expected_shapes[i])
        # Check that all weights are unique
        assert result[i].size == np.unique(result[i]).size


def test_forward_propagate():
    # A simple neural network that calculates XNOR expression for two binary inputs is used for the test
    X = np.asarray([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    weights = (np.asarray([[-30, 20, 20],
                           [10, -20, -20]]),
               np.asarray([-10, 20, 20]))
    expected_activations = ([[0, 1],
                             [0, 0],
                             [0, 0],
                             [1, 0]],
                            [1, 0, 0, 1])
    activation_function = SigmoidActivationFunction

    activations = BaseNeuralNetwork._forward_propagate(X, weights, activation_function)

    for i in range(len(expected_activations)):
        assert_allclose(activations[i], expected_activations[i], atol=1E-4)
