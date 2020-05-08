from abc import ABC
from mymllib._base import BaseSupervisedModel


class BaseRegression(BaseSupervisedModel, ABC):
    """Base class for regressions."""

    def _cost_gradient(self, params, X, y):
        # Intercept should not be regularized, so it is set to 0 in a copy of the parameters vector
        params_without_intercept = params.copy()
        params_without_intercept[0] = 0
        return (X.T@(self._hypothesis(X, params) - y) +
                self._regularization_param * params_without_intercept) / X.shape[0]
