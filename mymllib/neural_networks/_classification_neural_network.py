import numpy as np
from mymllib.neural_networks import BaseNeuralNetwork
from mymllib.preprocessing import one_hot
from mymllib.optimization import unroll
from mymllib.neural_networks.activations import Sigmoid
from mymllib.neural_networks.output_activations import SigmoidOutput, SoftmaxOutput
from mymllib.optimization import SciPyOptimizer


class ClassificationNeuralNetwork(BaseNeuralNetwork):
    """Feedforward fully connected neural network for classification problems.

    :param hidden_layers: Sizes of hidden layers of the neural network
    :param regularization_param: L2 regularization parameter (must be >= 0, when set exactly to 0 no regularization is
        used)
    :param optimizer: An optimizer to use for minimizing a cost function
    :param activation: Activation function for the neural network
    """

    def __init__(self, hidden_layers=(), regularization_param=0, optimizer=SciPyOptimizer("L-BFGS-B"),
                 activation=Sigmoid):
        super().__init__(hidden_layers=hidden_layers, regularization_param=regularization_param,
                         optimizer=optimizer, activation=activation)
        self._labels = None

    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        X, y = self._check_fit_data(X, y)
        self._labels, Y = one_hot(y)
        self._output_activation = SoftmaxOutput if Y.shape[1] > 2 else SigmoidOutput
        initial_weights = self._init_weights(X, Y)
        weights = self._optimize_params(X, Y, unroll(initial_weights))
        self._params = self._undo_weights_unroll(weights, X, Y)

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        X = self._check_data(X)

        predictions = super().predict(X)
        if len(self._labels) == 2:
            predictions = (predictions >= 0.5) * 1
        else:
            predictions = np.argmax(predictions, axis=1)
        return self._labels.take(predictions).flatten()
