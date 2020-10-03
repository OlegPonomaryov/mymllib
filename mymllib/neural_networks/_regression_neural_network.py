from mymllib.neural_networks import BaseNeuralNetwork
from mymllib.optimization import unroll
from mymllib.neural_networks.activations import Sigmoid
from mymllib.neural_networks.output_activations import IdentityOutput
from mymllib.optimization import SciPyOptimizer


class RegressionNeuralNetwork(BaseNeuralNetwork):
    """Feedforward fully connected neural network for regression problems.

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
        self._output_activation = IdentityOutput

    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        X, y = self._check_fit_data(X, y)
        y = y.reshape(-1, 1)
        initial_weights = self._init_weights(X, y)
        weights = self._optimize_params(X, y, unroll(initial_weights))
        self._params = self._undo_weights_unroll(weights, X, y)

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        X = self._check_data(X)
        return super().predict(X).flatten()
