import numpy as np
from scipy.special import comb


def to_numpy(a):
    """Convert input to a NumPy array.

    :param a: An object to convert
    :return: NumPy array
    """
    return np.asarray(a)


def add_intercept(X):
    """Add intercept feature (all-ones)

    :param X: Features
    :return: Features with intercept feature
    """
    samples_count = X.shape[0]
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
    X_result = np.zeros((X.shape[0], sum(poly_lengths)))
    X_result[:, :X.shape[1]] = X
    tail_index = X.shape[1]  # An index of the first column after the last polynomial inserted into the resulting matrix

    # Calculate all polynomials starting from 1 and up to the passed polynomial degree and insert them into the
    # resulting matrix
    for degree in range(2, polynomial_degree + 1):
        # Obtain the last added polynomial, its degree and length
        previous_poly_degree = degree - 1
        previous_poly_length = poly_lengths[previous_poly_degree - 1]
        previous_poly = X_result[:, tail_index-previous_poly_length:tail_index]

        # Iterate through column indices of the original matrix
        for i in range(0, X.shape[1]):
            # Calculate how many terms of the new polynomial should be calculated by multiplying terms of the previous
            # one by the i-th column of the original matrix
            terms_count = previous_poly_length if i == 0 else\
                          comb(X.shape[1] - i, previous_poly_degree, repetition=True, exact=True)

            # Calculate terms of the new polynomial that include the i-th column and append them to the resulting matrix
            X_result[:, tail_index:tail_index + terms_count] =\
                X[:, i][np.newaxis].T * previous_poly[:, previous_poly_length - terms_count:previous_poly_length]
            tail_index += terms_count

    return X_result
