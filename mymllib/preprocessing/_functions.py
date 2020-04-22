import numpy as np
from scipy.special import comb


def to_numpy(*args):
    """Convert input to a NumPy array.

    :param args: Objects to convert
    :return: Single NumPy array or a tuple of arrays (if multiple arguments passed)
    """
    if len(args) == 1:
        return np.asarray(args[0])
    else:
        return tuple(np.asarray(arg) for arg in args)


def add_intercept(X):
    """Add intercept feature (all-ones)

    :param X: Features
    :return: Features with intercept feature
    """
    samples_count = np.shape(X)[0]
    intercept_column = np.ones((samples_count, 1))
    return np.hstack((intercept_column, X))


def add_polynomial(X, polynomial_degree):
    """Add polynomial features (columns) to a matrix.

    :param X: The matrix to which polynomial columns should be added
    :param polynomial_degree: A polynomial degree up to which polynomial columns should be added (must be integer and
        greater than 0)
    :return: The matrix with added polynomial columns
    """
    if not isinstance(polynomial_degree, int):
        raise ValueError("Polynomial degree should be integer")
    if polynomial_degree < 1:
        raise ValueError("Polynomial degree should be greater than 0")

    X = to_numpy(X)

    # Calculate lengths of all polynomials with degrees starting from 1 and up to the passed polynomial degree
    poly_lengths = list(comb(X.shape[1], degree, repetition=True, exact=True)
                        for degree in range(1, polynomial_degree + 1))

    # Initialize resulting matrix and put the first degree polynomial (the original matrix) into it
    X_result = np.empty((X.shape[0], sum(poly_lengths)), dtype=X.dtype)
    X_result[:, :X.shape[1]] = X
    tail_index = X.shape[1]  # An index of the first column after the last polynomial inserted into the resulting matrix

    # Indices of positions from which to start multiplying terms of the previous polynomial by each column of the
    # original matrix to get the next polynomial
    start_indices = list(range(X.shape[1]))

    # Calculate all polynomials starting from 1 and up to the passed polynomial degree and insert them into the
    # resulting matrix
    for degree in range(2, polynomial_degree + 1):
        # Obtain the last added polynomial, its degree and length
        previous_poly_degree = degree - 1
        previous_poly_length = poly_lengths[previous_poly_degree - 1]
        previous_poly = X_result[:, tail_index-previous_poly_length:tail_index]

        # Iterate through column indices of the original matrix
        new_start_index = 0
        for i in range(0, X.shape[1]):
            # Calculate how many terms of the new polynomial should be calculated by multiplying terms of the previous
            # one by the i-th column of the original matrix
            terms_count = previous_poly_length - start_indices[i]

            # Calculate terms of the new polynomial that include the i-th column and append them to the resulting matrix
            np.multiply(previous_poly[:, previous_poly_length - terms_count:previous_poly_length], X[:, i, np.newaxis],
                        out=X_result[:, tail_index:tail_index + terms_count])

            tail_index += terms_count
            start_indices[i] = new_start_index
            new_start_index += terms_count

    return X_result


def one_hot(y):
    """One-hot encoding of labels sequence.

    :param y: Sequence of labels
    :return: Unique labels from the sequence, two-dimensional array of one-hot encoded labels
    """
    all_labels = np.unique(y)

    if len(all_labels) < 2:
        raise ValueError("There should be at least 2 different classes")

    labels = all_labels[1:] if len(all_labels) == 2 else all_labels
    return all_labels, np.vstack(tuple((y == label)*1 for label in labels)).T
