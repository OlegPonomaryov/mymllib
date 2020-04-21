"""Tests for the BaseNeuralNetwork class."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from mymllib.neural_networks import BaseNeuralNetwork
from mymllib.neural_networks.activation_functions import SigmoidActivationFunction
from mymllib.math.tools import gradient
from mymllib.optimization import unroll


@pytest.mark.parametrize("X, y, hidden_layers, expected_shapes", [
    (np.zeros((100, 5)), np.zeros((100, 3)), (), ((3, 6),)),
    (np.zeros((100, 16)), np.zeros((100, 9)), (27, 27), ((27, 17), (27, 28), (9, 28))),
    (np.zeros((100, 2)), np.zeros((100, 1)), (2,), ((2, 3), (1, 3)))
])
def test_init_weights(X, y, hidden_layers, expected_shapes):
    result = BaseNeuralNetwork._init_weights(X, y, hidden_layers=hidden_layers)

    for i in range(len(expected_shapes)):
        # Verify that weights matrix has correct shape
        assert_array_equal(result[i].shape, expected_shapes[i])
        # Check that all weights are unique
        assert result[i].size == np.unique(result[i]).size


X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
activation_function = SigmoidActivationFunction

# A simple neural network with a single output that calculates XNOR expression for two binary inputs
weights_so = (np.asarray([[-30, 20, 20],
                       [10, -20, -20]]),
           np.asarray([[-10, 20, 20]]))
expected_activations_so = (X,
                           [[0, 1],
                            [0, 0],
                            [0, 0],
                            [1, 0]],
                           [[1],
                            [0],
                            [0],
                            [1]])

# A simple neural network with multiple outputs that calculates XNOR and XOR expression for two binary inputs
weights_mo = (weights_so[0],
              np.asarray([[-10, 20, 20],
                          [10, -20, -20]]))
expected_activations_mo = (X,
                           expected_activations_so[1],
                           [[1, 0],
                            [0, 1],
                            [0, 1],
                            [1, 0]])


@pytest.mark.parametrize("X, weights, expected_activations", [
    (X, weights_so, expected_activations_so),
    (X, weights_mo, expected_activations_mo)
])
def test_forward_propagate(X, weights, expected_activations):
    activations = BaseNeuralNetwork._forward_propagate(X, weights, activation_function)

    for i in range(len(expected_activations)):
        assert_allclose(activations[i], expected_activations[i], atol=1E-4)


@pytest.mark.parametrize("X, y, weights", [
    (np.asarray(X), np.asarray(expected_activations_so[-1]), (np.ones((2, 3)), np.ones((1, 3)))),
    (np.asarray(X), np.asarray(expected_activations_so[-1]), (np.random.rand(2, 3)*2-1, np.random.rand(1, 3)*2-1)),
    (np.asarray(X), np.asarray(expected_activations_mo[-1]), (np.ones((2, 3)), np.ones((2, 3)))),
    (np.asarray(X), np.asarray(expected_activations_mo[-1]), (np.random.rand(2, 3)*2-1, np.random.rand(2, 3)*2-1)),
])
def test_cost_gradient(X, y, weights):
    unrolled_weights = unroll(weights)
    neural_network = BaseNeuralNetwork(hidden_layers=(2,), regularization_param=1, activation=activation_function)

    analytical_gradient = neural_network._cost_gradient(unrolled_weights, X, y)
    numerical_gradient = gradient(unrolled_weights, neural_network._cost, (X, y))

    assert_allclose(analytical_gradient, numerical_gradient)
