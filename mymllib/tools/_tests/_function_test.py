"""Tests for the '_functions' module."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from mymllib.tools import glorot_init


@pytest.mark.parametrize("shapes", [((3, 6),), ((3, 6), (5, 9))])
def test_glorot_init(shapes):
    weights = glorot_init(shapes,)

    for i in range(len(shapes)):
        # Verify that weights matrix has correct shape
        assert_array_equal(weights[i].shape, shapes[i])
        # Check that all weights are unique
        assert weights[i].size == np.unique(weights[i]).size
