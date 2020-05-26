import numpy as np
from mymllib import BaseSupervisedModel
from mymllib.preprocessing import add_intercept as add_bias
from mymllib.optimization import unroll, undo_unroll
from mymllib.neural_networks.activations import Sigmoid
from mymllib.optimization import LBFGSB
from mymllib.tools import glorot_init


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
        self._output_activation = None

    def _hypothesis(self, X, params):
        return BaseNeuralNetwork._forward_propagate(X, params, self._activation, self._output_activation)[-1]

    def _cost(self, params, X, y):
        weights = self._undo_weights_unroll(params, X, y)

        model_output = self._hypothesis(X, weights)
        loss_sum = np.sum(self._output_activation.loss(model_output, y))

        regularization = self._regularization_param / 2 * sum(np.sum(w[:, 1:]**2) for w in weights)

        return (loss_sum + regularization) / X.shape[0]

    def _cost_gradient(self, params, X, y):
        weights = self._undo_weights_unroll(params, X, y)
        activations = self._forward_propagate(X, weights, self._activation, self._output_activation)
        gradient = self._backpropagate(y, weights, activations, self._regularization_param, self._activation)
        return unroll(gradient)

    def _undo_weights_unroll(self, weights, X, y):
        shapes = BaseNeuralNetwork._get_weights_shapes(X, y, self._hidden_layers)
        return undo_unroll(weights, shapes)

    def _init_weights(self, X, y):
        # Weights are initialized with random values using Glorot initialization as it is implemented in scikit-learn -
        # with special factor value for sigmoid activation function
        factor = 2 if self._activation == Sigmoid else 6
        shapes = BaseNeuralNetwork._get_weights_shapes(X, y, self._hidden_layers)
        return glorot_init(shapes, factor)

    @staticmethod
    def _get_weights_shapes(X, y, hidden_layers):
        layers = (X.shape[1],) + hidden_layers + (y.shape[1],)
        return tuple((layers[i + 1], layers[i] + 1) for i in range(len(layers) - 1))

    @staticmethod
    def _forward_propagate(X, weights, activation_func, output_activation):
        net_activations = [add_bias(X)]

        # Calculate activations of the hidden layers
        for layer_weights in weights[:-1]:
            previous_layer_activations = net_activations[-1]
            layer_activations = activation_func.activations(previous_layer_activations @ layer_weights.T)
            net_activations.append(add_bias(layer_activations))

        # Calculate activations of the output layer
        output_layer_weights = weights[-1]
        previous_layer_activations = net_activations[-1]
        output_layer_activations = output_activation.activations(previous_layer_activations @ output_layer_weights.T)
        net_activations.append(output_layer_activations)

        return net_activations

    # noinspection PyPep8Naming
    @staticmethod
    def _backpropagate(y, weights, activations, regularization_param, activation_func):
        dL_dW = []
        dL_dz = activations[-1] - y
        for i in range(len(activations) - 2, -1, -1):
            dL_dW.insert(0, dL_dz.T @ activations[i])
            if i > 0:
                dL_da = dL_dz @ weights[i][:, 1:]
                da_dz = activation_func.derivative(activations[i][:, 1:])
                dL_dz = dL_da * da_dz

        dJ_dW = list()
        for i in range(len(dL_dW)):
            regularized_weights = regularization_param*weights[i]
            regularized_weights[:, 0] = 0
            dJ_dW.append((dL_dW[i] + regularized_weights) / y.shape[0])
        return dJ_dW
