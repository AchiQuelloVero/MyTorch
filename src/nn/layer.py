import numpy as np

from src.nn.activations import ACTIVATIONS
from src.nn.initializers import INITIALIZERS


class Dense:
    def __init__(self, input_size, size, activation, rng, init="xavier"):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'")
        if init not in INITIALIZERS:
            raise ValueError(f"unknown init '{init}'")
        if input_size < 1 or size < 1:
            raise ValueError("layer sizes must be positive")
        self.activation = activation
        self.init = init
        self.weights = INITIALIZERS[init](rng, input_size, size)
        self.biases = np.zeros(size)
        self.input = None
        self.output = None
        self.mask = 1.0
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    def forward(self, x, dropout=0, rng=None):
        self.input = x
        self.output = ACTIVATIONS[self.activation](x @ self.weights + self.biases)
        self.mask = (rng.random(self.output.shape) >= dropout) / (1 - dropout) if dropout else 1.0
        return self.output * self.mask

    def backward(self, delta, l2=0):
        self.grad_weights = self.input.T @ delta + l2 * self.weights
        self.grad_biases = delta.sum(axis=0)
        return delta @ self.weights.T
