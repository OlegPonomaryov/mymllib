import numpy as np
from mymllib import BaseModel


class NaiveBayesTextClassifier(BaseModel):
    """Naive Bayes with Laplacian smoothing for text classification."""

    def __init__(self):
        super().__init__()
        self._class_log_probs = None
        self._word_log_probs = None

    def fit(self, X, y):
        """Train the model.

        :param X: Features values
        :param y: Target values
        """
        X, y = self._check_fit_data(X, y)

        vocab_size = X.max() + 1
        num_classes = y.max() + 1
        samples_by_class = np.empty(num_classes)
        words_by_class = np.empty((vocab_size, num_classes))
        for c in range(num_classes):
            class_samples = X[y == c]
            samples_by_class[c] = class_samples.shape[0]
            for w in range(vocab_size):
                words_by_class[w, c] = (class_samples == w).sum()

        class_probs = samples_by_class / X.shape[0]
        word_probs = (words_by_class + 1) / (words_by_class.sum(axis=0) + vocab_size)

        self._class_log_probs = np.log(class_probs)
        self._word_log_probs = np.log(word_probs)

    def predict(self, X):
        """Predict target values.

        :param X: Features values
        :return: Target values
        """
        word_log_probs = self._word_log_probs[X]
        mask = np.stack([X >= 0] * 3, axis=-1)
        word_log_probs = np.where(mask, word_log_probs, np.zeros_like(word_log_probs))
        result_log_probs = self._class_log_probs + word_log_probs.sum(axis=1)
        return np.argmax(result_log_probs, axis=-1)
