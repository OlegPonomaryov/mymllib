import numpy as np


class BaseNeuralNetwork:
    """Base class for feedforward fully connected neural networks."""

    @staticmethod
    def _init_weights(input_layer, output_layer, hidden_layers=()):
        layers = (input_layer,) + hidden_layers + (output_layer,)

        if any(layer <= 0 for layer in layers):
            raise ValueError("Layer size should be greater than 0")

        init_epsilon = 1
        return tuple(np.random.rand(layers[i + 1], layers[i] + 1) * 2 * init_epsilon - init_epsilon
                     for i in range(len(layers) - 1))
