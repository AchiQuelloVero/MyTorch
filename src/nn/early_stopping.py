import numpy as np


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_layers = None

    def should_stop(self, epoch, loss, layers):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_layers = [(layer.weights.copy(), layer.biases.copy()) for layer in layers]
            return False
        return epoch - self.best_epoch >= self.patience

    def restore(self, layers):
        if self.best_layers is None:
            return
        for layer, (weights, biases) in zip(layers, self.best_layers):
            layer.weights = weights
            layer.biases = biases
