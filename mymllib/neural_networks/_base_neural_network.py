import numpy as np
from mymllib import BaseSupervisedModel
from mymllib.preprocessing import add_intercept
from mymllib.optimization import unroll, undo_unroll
from mymllib.neural_networks.activation_functions import SigmoidActivationFunction


class BaseNeuralNetwork(BaseSupervisedModel):
    """Base class for feedforward fully connected neural networks."""

    def __init__(self, hidden_layers=(), regularization_param=0, activation=SigmoidActivationFunction):
        super().__init__(regularization_param=regularization_param)
        self._hidden_layers = hidden_layers
        self._activation = activation

    def _hypothesis(self, X, params):
        return BaseNeuralNetwork._forward_propagate(X, params, self._activation)[-1]

    def _cost(self, params, X, y):
        weights = self._undo_weights_unroll(params, X, y)

        log_loss = y*np.log(self._hypothesis(X, weights)) + (1 - y)*np.log(1 - self._hypothesis(X, weights))
        log_loss = -np.sum(log_loss) / X.shape[0]

        regularization = self._regularization_param / (2 * X.shape[0]) *\
            sum((layer_weights[:, 1:] ** 2).sum() for layer_weights in weights)
        return log_loss + regularization

    def _cost_gradient(self, params, X, y):
        weights = self._undo_weights_unroll(params, X, y)
        activations = self._forward_propagate(X, weights, self._activation)
        gradient = self._backpropagate(y, weights, activations, self._regularization_param, self._activation)
        return unroll(gradient)

    def _undo_weights_unroll(self, weights, X, y):
        shapes = BaseNeuralNetwork._get_weights_shapes(X, y, self._hidden_layers)
        return undo_unroll(weights, shapes)

    @staticmethod
    def _init_weights(X, y, hidden_layers):
        shapes = BaseNeuralNetwork._get_weights_shapes(X, y, hidden_layers)
        init_epsilon = 1
        return tuple(np.random.rand(shape[0], shape[1]) * 2 * init_epsilon - init_epsilon for shape in shapes)

    @staticmethod
    def _get_weights_shapes(X, y, hidden_layers):
        layers = (X.shape[1],) + hidden_layers + (y.shape[1],)
        return tuple((layers[i + 1], layers[i] + 1) for i in range(len(layers) - 1))

    @staticmethod
    def _forward_propagate(X, weights, activation_func):
        activations = [X]
        for layer_weights in weights:
            previous_activations = add_intercept(activations[-1])
            activations.append(activation_func.activations(previous_activations @ layer_weights.T))
        return activations

    @staticmethod
    def _backpropagate(y, weights, activations, regularization_param, activation_func):
        D = []
        d = activations[-1] - y
        for i in range(len(activations) - 2, -1, -1):
            D.insert(0, d.T @ add_intercept(activations[i]))
            if i > 0:
                z = add_intercept(activations[i - 1]) @ weights[i - 1].T
                d = d @ weights[i][:, 1:] * activation_func.derivative(z)
        for i in range(len(D)):
            regularized_weights = regularization_param*weights[i]
            regularized_weights[:, 0] = 0
            D[i] = (D[i] + regularized_weights) / y.shape[0]
        return D
