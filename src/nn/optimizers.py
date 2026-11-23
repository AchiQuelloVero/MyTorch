import numpy as np


def parameters(layers):
    pairs = []
    for layer in layers:
        pairs.append((layer.weights, layer.grad_weights))
        pairs.append((layer.biases, layer.grad_biases))
    return pairs


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers):
        for param, grad in parameters(layers):
            param -= self.learning_rate * grad


class Momentum:
    def __init__(self, learning_rate, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = []

    def step(self, layers):
        pairs = parameters(layers)
        if not self.velocity:
            self.velocity = [np.zeros_like(param) for param, _ in pairs]
        for (param, grad), velocity in zip(pairs, self.velocity):
            velocity *= self.beta
            velocity -= self.learning_rate * grad
            param += velocity


class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0

    def step(self, layers):
        pairs = parameters(layers)
        if not self.m:
            self.m = [np.zeros_like(param) for param, _ in pairs]
            self.v = [np.zeros_like(param) for param, _ in pairs]
        self.t += 1
        for (param, grad), m, v in zip(pairs, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def make_optimizer(config):
    name = config.get("optimizer", "sgd")
    learning_rate = float(config.get("learning_rate", 0.01))
    if name == "sgd":
        return SGD(learning_rate)
    if name == "momentum":
        return Momentum(learning_rate, float(config.get("momentum", 0.9)))
    if name == "adam":
        return Adam(learning_rate, float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999)))
    raise ValueError(f"invalid config: unknown optimizer '{name}'")
