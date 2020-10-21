import numpy as np
from ._common import _check_input


def precision(y_actual, y_predicted, target_label=1):
    """Calculate the precision of a prediction.

    :param y_actual: Actual labels
    :param y_predicted: Predicted labels
    :param target_label: A target label for which to calculate the precision
    :return: Precision value
    """
    y_actual, y_predicted = _check_input(y_actual, y_predicted)

    return _precision(y_actual, y_predicted, target_label)


def recall(y_actual, y_predicted, target_label=1):
    """Calculate the recall of a prediction.

    :param y_actual: Actual labels
    :param y_predicted: Predicted labels
    :param target_label: A target label for which to calculate the recall
    :return: Recall value
    """
    y_actual, y_predicted = _check_input(y_actual, y_predicted)

    return _recall(y_actual, y_predicted, target_label)


def f1_score(y_actual, y_predicted, target_label=1, use_target_for_multiclass=False):
    """Calculate the F1 score a prediction.

    :param y_actual: Actual labels
    :param y_predicted: Predicted labels
    :param target_label: A target label for which to calculate the F1 score
    :param use_target_for_multiclass: Whether to calculate F1 score with respect only to the target label for multiclass
        problems (otherwise the target label will be ignored and weighted average score for all labels will be returned)
    :return: F1 score value
    """
    y_actual, y_predicted = _check_input(y_actual, y_predicted)

    labels = np.unique(y_actual)
    calculate_weighted = len(labels) > 2 and not use_target_for_multiclass

    total_f1_score = 0
    target_labels = labels if calculate_weighted else (target_label,)

    for label in target_labels:
        precision = _precision(y_actual, y_predicted, label)
        recall = _recall(y_actual, y_predicted, label)
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        if calculate_weighted:
            f1_score *= np.count_nonzero(y_actual == label)
        total_f1_score += f1_score

    if calculate_weighted:
        total_f1_score /= len(y_actual)

    return total_f1_score


# Precision calculation without converting of parameters to NumPy arrays (to avoid multiple conversions of the same
# parameter when called from f1_score())
def _precision(y_actual, y_predicted, target_label):
    predicted_positive = y_predicted == target_label
    predicted_positive_count = np.count_nonzero(predicted_positive)

    if predicted_positive_count == 0:
        return 0

    true_positive = y_actual[predicted_positive] == target_label
    return np.count_nonzero(true_positive) / predicted_positive_count


# Recall calculation without converting of parameters to NumPy arrays (to avoid multiple conversions of the same
# parameter when called from f1_score())
def _recall(y_actual, y_predicted, target_label):
    actual_positive = y_actual == target_label
    actual_positive_count = np.count_nonzero(actual_positive)

    if actual_positive_count == 0:
        return 0

    true_positive = y_predicted[actual_positive] == target_label
    return np.count_nonzero(true_positive) / actual_positive_count
