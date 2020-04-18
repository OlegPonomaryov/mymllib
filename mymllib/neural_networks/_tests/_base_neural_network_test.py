"""Tests for the BaseNeuralNetwork class."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from mymllib.neural_networks import BaseNeuralNetwork


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
