import numpy as np


class HiddenMarkov:
    """A Hidden Markov model implementation with Viterbi algorithm for optimal path selection.

    :param e: A value used for smoothing
    """

    def __init__(self, e=0.001):
        self.e = e
        self.transition_matrix = None
        self.emission_matrix = None

    def fit(self, X, Y):
        """Train the model.

        :param X: Observable states
        :param Y: Hidden states
        """
        num_observable_states = max(max(x) for x in X) + 1
        num_hidden_states = max(max(y) for y in Y) + 1

        self.transition_matrix = self._transition_matrix(Y, num_hidden_states)
        self.emission_matrix = self._emission_matrix(X, num_observable_states,
                                                     Y, num_hidden_states)

    def predict(self, X):
        """Predict hidden states.

        :param X: Observable states
        :return: Hidden states
        """
        return [self._viterbi(x) for x in X]

    def _transition_matrix(self, Y, Y_size):
        mat = np.zeros((Y_size + 1, Y_size))
        for y in Y:
            mat[0, y[0]] += 1
            for i in range(1, len(y)):
                mat[y[i - 1] + 1, y[i]] += 1
        # Smoothing is not applied to transitions from the initial states, since some state might never be in th first
        # position (i.e. some parts of speech can never be a first word of a sentence)
        mat[1:] += self.e
        for i in range(mat.shape[0]):
            mat[i] /= mat[i].sum()
        return mat

    def _emission_matrix(self, X, X_size, Y, Y_size):
        mat = np.zeros((Y_size, X_size))
        for x, y in zip(X, Y):
            for i in range(len(x)):
                mat[y[i], x[i]] += 1
        mat += self.e
        for i in range(mat.shape[0]):
            mat[i] /= mat[i].sum()
        return mat

    def _viterbi(self, x):
        # INIT
        x_size = len(x)
        num_states = self.emission_matrix.shape[0]
        intermediate_probs = np.empty((num_states, x_size))
        intermediate_states = np.empty((num_states, x_size - 1), dtype=int)

        # FORWARD PASS
        # Since transitions from the initial state weren't smoothed, they might contain zeros, that are replaced with a
        # very small value to avoid errors when calculating logarithm
        intermediate_probs[:, 0] = \
            np.log(np.max(self.transition_matrix[0] + 1e-300)) + np.log(self.emission_matrix[:, x[0]])
        for i in range(1, x_size):
            log_probs = intermediate_probs[:, i - 1][:, np.newaxis] + \
                        np.log(self.transition_matrix[1:]) + \
                        np.log(self.emission_matrix[:, x[i]])
            intermediate_probs[:, i] = log_probs.max(axis=0)
            intermediate_states[:, i - 1] = log_probs.argmax(axis=0)

        # BACKWARD PASS
        y = [np.argmax(intermediate_probs[:, -1])]
        for i in range(x_size - 2, -1, -1):
            y.insert(0, intermediate_states[y[0], i])

        return y
