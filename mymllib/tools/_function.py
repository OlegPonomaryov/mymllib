from math import sqrt
import numpy as np


def glorot_init(shapes, factor=6):
    weights = []
    for shape in shapes:
        init_epsilon = sqrt(factor / (shape[0] + shape[1]))
        layer_weights = np.random.rand(shape[0], shape[1]) * 2 * init_epsilon - init_epsilon
        weights.append(layer_weights)
    return tuple(weights)
