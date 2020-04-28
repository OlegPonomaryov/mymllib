import numpy as np
from math import sqrt
from mymllib import BaseSupervisedModel
from mymllib.preprocessing import add_intercept as add_bias
from mymllib.optimization import unroll, undo_unroll
from mymllib.neural_networks.activations import Sigmoid
from mymllib.optimization import LBFGSB


class BaseNeuralNetwork(BaseSupervisedModel):
    """Base class for feedforward fully connected neural networks.

    :param hidden_layers: Sizes of hidden layers of the neural network
    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function
    :param activation: Activation function for the neural network
    """

    def __init__(self, hidden_layers=(), regularization_param=0, optimizer=LBFGSB(), activation=Sigmoid):
        super().__init__(regularization_param=regularization_param, optimizer=optimizer)
        self._hidden_layers = hidden_layers
        self._activation = activation

    def _hypothesis(self, X, params):
        return BaseNeuralNetwork._forward_propagate(X, params, self._activation)[-1]

    def _cost(self, params, X, y):
        weights = self._undo_weights_unroll(params, X, y)

        model_output = self._hypothesis(X, weights)
        log_loss = y*np.log(model_output) + (1 - y)*np.log(1 - model_output)
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

    def _init_weights(self, X, y):
        # Weights are initialized with random values using Glorot initialization (as it is implemented in scikit-learn -
        # with special factor value for sigmoid activation function)
        factor = 2 if self._activation == Sigmoid else 6
        shapes = BaseNeuralNetwork._get_weights_shapes(X, y, self._hidden_layers)
        weights = []
        for shape in shapes:
            init_epsilon = sqrt(factor / (shape[0] + shape[1]))
            layer_weights = np.random.rand(shape[0], shape[1]) * 2 * init_epsilon - init_epsilon
            weights.append(layer_weights)
        return weights

    @staticmethod
    def _get_weights_shapes(X, y, hidden_layers):
        layers = (X.shape[1],) + hidden_layers + (y.shape[1],)
        return tuple((layers[i + 1], layers[i] + 1) for i in range(len(layers) - 1))

    @staticmethod
    def _forward_propagate(X, weights, activation_func):
        activations = []
        previous_activations = X
        for layer_weights in weights:
            previous_activations = add_bias(previous_activations)
            activations.append(previous_activations)
            previous_activations = activation_func.activations(previous_activations @ layer_weights.T)
        activations.append(previous_activations)  # Add last layer's activations (network's output) without a bias
        return activations

    @staticmethod
    def _backpropagate(y, weights, activations, regularization_param, activation_func):
        D = []
        d = activations[-1] - y
        for i in range(len(activations) - 2, -1, -1):
            D.insert(0, d.T @ activations[i])
            if i > 0:
                d = d @ weights[i][:, 1:] * activation_func.derivative(activations[i][:, 1:])
        for i in range(len(D)):
            regularized_weights = regularization_param*weights[i]
            regularized_weights[:, 0] = 0
            D[i] = (D[i] + regularized_weights) / y.shape[0]
        return D
