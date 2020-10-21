"""Tests for the 'regression' module."""
import pytest
import numpy as np
from mymllib.metrics.regression import mean_absolute_error, mean_absolute_percentage_error
from mymllib.preprocessing import to_numpy


@pytest.mark.parametrize("func", [mean_absolute_error, mean_absolute_percentage_error])
@pytest.mark.parametrize("actual, predicted", [
    (np.ones(5), np.ones((5, 1))),
    (np.ones((5, 1)), np.ones(5)),
    (np.ones((5, 1)), np.ones((5, 1)))])
def test_mae_mape__invalid_input_dimensions(func, actual, predicted):
    with pytest.raises(ValueError):
        func(actual, predicted)


@pytest.mark.parametrize("func", [mean_absolute_error, mean_absolute_percentage_error])
@pytest.mark.parametrize("actual, predicted", [
    (np.ones(5), np.ones(4)),
    (np.ones(4), np.ones(5))])
def test_mae_mape____input_shapes_mismatch(func, actual, predicted):
    with pytest.raises(ValueError):
        func(actual, predicted)


def test_mean_absolute_error():
    actual_values = to_numpy([13, 122.5, 18, -20])
    predicted_values = to_numpy([12.5, 124, 19, -19])
    expected_mae = 1

    mae = mean_absolute_error(actual_values, predicted_values)

    assert mae == expected_mae


def test_mean_absolute_percentage_error():
    actual_values = to_numpy([20, 1, -5, 10])
    predicted_values = to_numpy([19, 0.5, -6, 10.5])
    expected_mape = 20

    mape = mean_absolute_percentage_error(actual_values, predicted_values)

    assert mape == expected_mape
