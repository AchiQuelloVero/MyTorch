import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sigmoid_prime(a):
    return a * (1 - a)


def tanh_prime(a):
    return 1 - a ** 2


def relu_prime(a):
    return (a > 0).astype(float)


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "softmax": softmax,
}

DERIVATIVES = {
    "sigmoid": sigmoid_prime,
    "tanh": tanh_prime,
    "relu": relu_prime,
}
