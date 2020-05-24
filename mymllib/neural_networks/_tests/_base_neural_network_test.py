"""Tests for the BaseNeuralNetwork class."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_almost_equal
from mymllib.neural_networks import BaseNeuralNetwork
from mymllib.neural_networks.activations import Sigmoid
from mymllib.neural_networks.output_activations import SigmoidOutput, SoftmaxOutput
from mymllib.math.tools import gradient
from mymllib.optimization import unroll
from mymllib.preprocessing import add_intercept, one_hot


@pytest.mark.parametrize("X, y, hidden_layers, expected_shapes", [
    (np.zeros((100, 5)), np.zeros((100, 3)), (), ((3, 6),)),
    (np.zeros((100, 16)), np.zeros((100, 9)), (27, 27), ((27, 17), (27, 28), (9, 28))),
    (np.zeros((100, 2)), np.zeros((100, 1)), (2,), ((2, 3), (1, 3)))
])
def test_init_weights(X, y, hidden_layers, expected_shapes):
    neural_network = BaseNeuralNetwork(hidden_layers=hidden_layers)
    result = neural_network._init_weights(X, y)

    for i in range(len(expected_shapes)):
        # Verify that weights matrix has correct shape
        assert_array_equal(result[i].shape, expected_shapes[i])
        # Check that all weights are unique
        assert result[i].size == np.unique(result[i]).size


X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
activation_function = Sigmoid

# A simple neural network with a single output that calculates XNOR expression for two binary inputs
weights_so = (np.asarray([[-30, 20, 20],
                       [10, -20, -20]]),
           np.asarray([[-10, 20, 20]]))
expected_activations_so = (add_intercept(X),
                           add_intercept(
                           [[0, 1],
                            [0, 0],
                            [0, 0],
                            [1, 0]]),
                           [[1],
                            [0],
                            [0],
                            [1]])

# A simple neural network with multiple outputs that calculates XNOR and XOR expression for two binary inputs
weights_mo = (weights_so[0],
              np.asarray([[-10, 20, 20],
                          [10, -20, -20]]))
expected_activations_mo = (expected_activations_so[0],
                           expected_activations_so[1],
                           [[1, 0],
                            [0, 1],
                            [0, 1],
                            [1, 0]])


@pytest.mark.parametrize("X, weights, output_activation, expected_activations", [
    (X, weights_so, SigmoidOutput, expected_activations_so),
    (X, weights_mo, SigmoidOutput, expected_activations_mo),
    (X, weights_mo, SoftmaxOutput, expected_activations_mo)
])
def test_forward_propagate(X, weights, output_activation, expected_activations):
    activations = BaseNeuralNetwork._forward_propagate(X, weights, activation_function, output_activation)

    for i in range(len(expected_activations)):
        assert_allclose(activations[i], expected_activations[i], atol=1E-4)


@pytest.mark.parametrize("X, y, weights, output_activation", [
    (np.asarray(X), np.asarray(expected_activations_so[-1]), (np.ones((2, 3)), np.ones((1, 3))), SigmoidOutput),
    (np.asarray(X), np.asarray(expected_activations_so[-1]), (np.random.rand(2, 3)*2-1, np.random.rand(1, 3)*2-1),
     SigmoidOutput),
    (np.asarray(X), np.asarray(expected_activations_mo[-1]), (np.ones((2, 3)), np.ones((2, 3))), SigmoidOutput),
    (np.asarray(X), np.asarray(expected_activations_mo[-1]), (np.random.rand(2, 3)*2-1, np.random.rand(2, 3)*2-1),
     SigmoidOutput),
    (np.asarray(X), np.asarray(expected_activations_mo[-1]), (np.ones((2, 3)), np.ones((2, 3))), SoftmaxOutput),
    (np.asarray(X), np.asarray(expected_activations_mo[-1]), (np.random.rand(2, 3)*2-1, np.random.rand(2, 3)*2-1),
     SoftmaxOutput)
])
def test_cost_gradient(X, y, weights, output_activation):
    unrolled_weights = unroll(weights)
    neural_network = BaseNeuralNetwork(hidden_layers=(2,), regularization_param=1, activation=activation_function)
    neural_network._output_activation = output_activation

    analytical_gradient = neural_network._cost_gradient(unrolled_weights, X, y)
    numerical_gradient = gradient(unrolled_weights, neural_network._cost, (X, y))

    assert_almost_equal(analytical_gradient, numerical_gradient)


@pytest.mark.parametrize("samples_count, features_count", [(5, 10)])
@pytest.mark.parametrize("classes_count, output_activation", [
    (2, SigmoidOutput), (3, SigmoidOutput), (3, SoftmaxOutput)])
@pytest.mark.parametrize("hidden_layers", [(5,), (10,), (20,), (10, 10, 10)])
def test_cost_gradient__random_input(samples_count, features_count, classes_count, hidden_layers, output_activation):
    random_state = np.random.RandomState(seed=7)
    X = np.asarray(random_state.rand(samples_count, features_count))
    y = one_hot(1 + np.mod(np.arange(samples_count) + 1, classes_count))[1]

    neural_network = BaseNeuralNetwork(hidden_layers=hidden_layers)
    neural_network._output_activation = output_activation
    initial_weights = neural_network._init_weights(X, y)
    weights = neural_network._optimize_params(X, y, unroll(initial_weights))

    analytical_gradient = neural_network._cost_gradient(weights, X, y)
    numerical_gradient = gradient(weights, neural_network._cost, (X, y))

    assert_almost_equal(analytical_gradient, numerical_gradient)
