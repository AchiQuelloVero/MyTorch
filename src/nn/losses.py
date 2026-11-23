import numpy as np

from src.nn.activations import DERIVATIVES

EPSILON = 1e-12


def weighted_mean(values, weights):
    if weights is None:
        return np.mean(values)
    return np.sum(values * weights) / np.sum(weights)


def cross_entropy(output, expected, weights=None):
    output = np.clip(output, EPSILON, 1)
    return weighted_mean(-np.sum(expected * np.log(output), axis=1), weights)


def binary_cross_entropy(output, expected, weights=None):
    output = np.clip(output, EPSILON, 1 - EPSILON)
    cost = expected * np.log(output) + (1 - expected) * np.log(1 - output)
    return weighted_mean(-np.sum(cost, axis=1), weights)


def mse(output, expected, weights=None):
    return weighted_mean(np.sum((output - expected) ** 2, axis=1), weights)


LOSSES = {
    "cross_entropy": cross_entropy,
    "binary_cross_entropy": binary_cross_entropy,
    "mse": mse,
}

OUTPUT_ACTIVATIONS = {
    "cross_entropy": ("softmax",),
    "binary_cross_entropy": ("sigmoid",),
    "mse": ("sigmoid", "tanh", "relu"),
}


def output_delta(loss, activation, output, expected, weights=None):
    if loss == "mse":
        delta = 2 * (output - expected) * DERIVATIVES[activation](output)
    else:
        delta = output - expected
    if weights is None:
        return delta / len(output)
    return delta * weights[:, None] / np.sum(weights)
