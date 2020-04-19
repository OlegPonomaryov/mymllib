import numpy as np
from mymllib.preprocessing import add_intercept


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

    @staticmethod
    def _forward_propagate(X, weights, activation_func):
        activations = []
        previous_activations = X
        for layer_weights in weights:
            previous_activations = add_intercept(previous_activations)
            activations.append(activation_func.activations(previous_activations, layer_weights))
            previous_activations = activations[-1]
        return activations

