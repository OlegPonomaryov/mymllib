"""Tests for the 'classification' module."""
import pytest
import numpy as np
from mymllib.metrics.classification import precision, recall, f1_score


actual_values = ["A", "C", "B", "C"]
predicted_values = ["B", "D", "B", "C"]


@pytest.mark.parametrize("func", [precision, recall, f1_score])
@pytest.mark.parametrize("actual, predicted", [
    (np.ones(5), np.ones((5, 1))),
    (np.ones((5, 1)), np.ones(5)),
    (np.ones((5, 1)), np.ones((5, 1)))])
def test_precision_recall_f1_score__invalid_input_dimensions(func, actual, predicted):
    with pytest.raises(ValueError):
        func(actual, predicted, target_label=1)


@pytest.mark.parametrize("func", [precision, recall, f1_score])
@pytest.mark.parametrize("actual, predicted", [
    (np.ones(5), np.ones(4)),
    (np.ones(4), np.ones(5))])
def test_precision_recall_f1_score__input_shapes_mismatch(func, actual, predicted):
    with pytest.raises(ValueError):
        func(actual, predicted, target_label=1)


def test_precision__target_label_not_in_predicted_values():
    with pytest.raises(ValueError):
        precision(actual_values, predicted_values, target_label="A")


@pytest.mark.parametrize("target_label, expectation", [("B", 0.5), ("C", 1), ("D", 0)])
def test_precision(target_label, expectation):
    p = precision(actual_values, predicted_values, target_label=target_label)

    assert p == expectation


def test_recall__target_label_not_in_actual_values():
    with pytest.raises(ValueError):
        recall(actual_values, predicted_values, target_label="D")


@pytest.mark.parametrize("target_label, expectation", [("A", 0), ("B", 1), ("C", 0.5)])
def test_recall(target_label, expectation):
    r = recall(actual_values, predicted_values, target_label=target_label)

    assert r == expectation


def test_f1_score():
    actual_values = ["A", "B", "B", "A", "B", "C"]
    predicted_values = ["A", "A", "B", "B", "B", "C"]
    expected_f1 = (0.5*2 + 2/3*3 + 1)/6

    f1 = f1_score(actual_values, predicted_values, use_target_for_multiclass=False)

    assert f1 == expected_f1


@pytest.mark.parametrize("target_label", ["A", "D"])
def test_f1_score__use_target_for_multiclass__target_label_not_in_predicted_or_actual_values(target_label):
    with pytest.raises(ValueError):
        f1_score(actual_values, predicted_values, target_label=target_label, use_target_for_multiclass=True)


@pytest.mark.parametrize("target_label, expectation", [("B", 1/1.5), ("C", 1/1.5)])
def test_f1_score__use_target_for_multiclass(target_label, expectation):
    f1 = f1_score(actual_values, predicted_values, target_label=target_label, use_target_for_multiclass=True)

    assert f1 == expectation
