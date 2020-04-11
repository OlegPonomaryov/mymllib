"""Tests for optimizers (subclasses of the BaseOptimizer class)."""
import pytest
from numpy.testing import assert_allclose
from mymllib.optimization import GradientDescent, LBFGSB
from mymllib.preprocessing import to_numpy


def f(x):
    return 2*x[0]**2 + 0.5*x[1]**4 + 130*x[2]**2


def grad_f(x):
    return to_numpy([4*x[0], 2*x[1], 260*x[2]])


min_f = to_numpy([0, 0, 0])  # Minimum of f()


@pytest.mark.parametrize("optimizer", [GradientDescent(), LBFGSB()])
def test_fit_predict(optimizer):
    x0 = to_numpy([-7, 15, 4])
    x = optimizer.minimize(f, grad_f, x0)
    assert_allclose(x, min_f, atol=1E-5)
