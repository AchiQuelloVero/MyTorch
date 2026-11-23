import json
import sys

import numpy as np

from src.nn.activations import DERIVATIVES
from src.nn.early_stopping import EarlyStopping
from src.nn.layer import Dense
from src.nn.losses import LOSSES, OUTPUT_ACTIVATIONS, output_delta
from src.nn.optimizers import make_optimizer

CHUNK = 4096
FORMAT = "my_torch_network"
VERSION = 1


class Network:
    def __init__(self, config):
        try:
            self.config = config
            self.input_size = int(config["input_size"])
            self.classes = list(config["classes"])
            self.loss = config.get("loss", "cross_entropy")
            self.learning_rate = float(config.get("learning_rate", 0.01))
            self.lr_decay = float(config.get("lr_decay", 0))
            self.batch_size = int(config.get("batch_size", 32))
            self.epochs = int(config.get("epochs", 10))
            self.early_stopping = int(config.get("early_stopping", 0))
            self.l2 = float(config.get("l2", 0))
            self.dropout = float(config.get("dropout", 0))
            self.optimizer = make_optimizer(config)
            init = config.get("init", "xavier")
            self.rng = np.random.default_rng(config.get("seed"))
            self.layers = []
            size = self.input_size
            for layer in config["layers"]:
                dense = Dense(size, int(layer["size"]), layer["activation"], self.rng, layer.get("init", init))
                self.layers.append(dense)
                size = int(layer["size"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"invalid config: {e}")
        if not self.layers:
            raise ValueError("invalid config: no layers")
        if size != len(self.classes):
            raise ValueError(f"invalid config: {size} outputs for {len(self.classes)} classes")
        self.check_activations()
        self.check_training()

    def check_activations(self):
        for layer in self.layers[:-1]:
            if layer.activation not in DERIVATIVES:
                raise ValueError(f"invalid config: '{layer.activation}' can only be the output activation")
        if self.loss not in LOSSES:
            raise ValueError(f"invalid config: unknown loss '{self.loss}'")
        output = self.layers[-1].activation
        if output not in OUTPUT_ACTIVATIONS[self.loss]:
            raise ValueError(f"invalid config: '{self.loss}' does not work with a '{output}' output")

    def check_training(self):
        if self.learning_rate <= 0:
            raise ValueError("invalid config: learning_rate must be positive")
        if self.lr_decay < 0:
            raise ValueError("invalid config: lr_decay can't be negative")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("invalid config: batch_size and epochs must be at least 1")
        if self.early_stopping < 0:
            raise ValueError("invalid config: early_stopping can't be negative")
        if self.l2 < 0:
            raise ValueError("invalid config: l2 can't be negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("invalid config: dropout must be at least 0 and less than 1")

    @classmethod
    def from_config(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def save(self, path):
        data = {
            "format": FORMAT,
            "version": VERSION,
            "config": self.config,
            "classes": self.classes,
            "input_size": self.input_size,
            "layers": [
                {"activation": layer.activation, "weights": layer.weights.tolist(), "biases": layer.biases.tolist()}
                for layer in self.layers
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: not a JSON file ({e})")
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        if data.get("version") != VERSION:
            raise ValueError(f"{path}: unsupported version {data.get('version')}")
        try:
            net = cls(data["config"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"{path}: {e}")
        if data.get("classes") != net.classes or data.get("input_size") != net.input_size:
            raise ValueError(f"{path}: classes or input_size don't match the config")
        layers = data.get("layers")
        if not isinstance(layers, list) or len(layers) != len(net.layers):
            raise ValueError(f"{path}: expected {len(net.layers)} layers")
        for number, (layer, saved) in enumerate(zip(net.layers, layers), 1):
            load_layer(layer, saved, f"{path}: layer {number}")
        return net

    def forward(self, x, training=False):
        x = np.asarray(x, dtype=float)
        if x.shape[1] != self.input_size:
            raise ValueError(f"expected {self.input_size} inputs, got {x.shape[1]}")
        dropout = self.dropout if training else 0
        for layer in self.layers[:-1]:
            x = layer.forward(x, dropout, self.rng)
        return self.layers[-1].forward(x)

    def predict(self, x):
        x = np.atleast_2d(x)
        if len(x) == 0:
            return np.empty((0, len(self.classes)))
        return np.concatenate([self.forward(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)])

    def targets(self, expected):
        expected = np.asarray(expected)
        if expected.ndim == 1 and np.issubdtype(expected.dtype, np.integer):
            if expected.size and (expected.min() < 0 or expected.max() >= len(self.classes)):
                raise ValueError(f"class indices must be between 0 and {len(self.classes) - 1}")
            return np.eye(len(self.classes))[expected]
        expected = np.atleast_2d(np.asarray(expected, dtype=float))
        if expected.shape[1] != len(self.classes):
            raise ValueError(f"expected {len(self.classes)} outputs, got {expected.shape[1]}")
        return expected

    def penalty(self):
        return 0.5 * self.l2 * sum(np.sum(layer.weights ** 2) for layer in self.layers)

    def cost(self, x, expected, weights=None):
        return LOSSES[self.loss](self.predict(x), self.targets(expected), weights) + self.penalty()

    def evaluate(self, x, expected):
        output = self.predict(x)
        expected = self.targets(expected)
        accuracy = np.mean(output.argmax(axis=1) == expected.argmax(axis=1))
        return float(LOSSES[self.loss](output, expected)), float(accuracy)

    def backward(self, expected, weights=None):
        last = self.layers[-1]
        delta = output_delta(self.loss, last.activation, last.output, expected, weights)
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(delta, self.l2)
            if i > 0:
                previous = self.layers[i - 1]
                delta = grad * previous.mask * DERIVATIVES[previous.activation](previous.output)

    def train_step(self, x, expected, weights=None):
        expected = self.targets(expected)
        cost = LOSSES[self.loss](self.forward(np.atleast_2d(x), True), expected, weights)
        self.backward(expected, weights)
        self.optimizer.step(self.layers)
        return cost

    def train(self, x, expected, x_val=None, y_val=None, class_weights=None, verbose=False):
        x = np.atleast_2d(x)
        expected = self.targets(expected)
        if len(x) != len(expected):
            raise ValueError(f"{len(x)} inputs for {len(expected)} expected outputs")
        weights = self.sample_weights(expected, class_weights)
        stopper = EarlyStopping(self.early_stopping) if self.early_stopping and x_val is not None else None
        history = []
        for epoch in range(self.epochs):
            self.optimizer.learning_rate = self.learning_rate / (1 + self.lr_decay * epoch)
            order = self.rng.permutation(len(x))
            total, correct = 0.0, 0
            for start in range(0, len(x), self.batch_size):
                batch = order[start:start + self.batch_size]
                batch_weights = None if weights is None else weights[batch]
                total += self.train_step(x[batch], expected[batch], batch_weights) * len(batch)
                correct += np.sum(self.layers[-1].output.argmax(axis=1) == expected[batch].argmax(axis=1))
            record = {"epoch": epoch + 1, "loss": float(total / len(x)), "accuracy": float(correct / len(x))}
            if x_val is not None:
                record["val_loss"], record["val_accuracy"] = self.evaluate(x_val, y_val)
            history.append(record)
            if verbose:
                print(describe(record, self.epochs), file=sys.stderr)
            if stopper and stopper.should_stop(epoch + 1, record["val_loss"], self.layers):
                break
        if stopper:
            stopper.restore(self.layers)
            if verbose and stopper.best_epoch:
                print(f"best network: epoch {stopper.best_epoch} of {len(history)} "
                      f"(val_loss {stopper.best_loss:.4f})", file=sys.stderr)
        return history

    def sample_weights(self, expected, class_weights):
        if class_weights is None:
            return None
        class_weights = np.asarray(class_weights, dtype=float)
        if class_weights.shape != (len(self.classes),):
            raise ValueError(f"expected {len(self.classes)} class weights, got {class_weights.size}")
        return class_weights[expected.argmax(axis=1)]


def load_layer(layer, saved, where):
    try:
        activation = saved["activation"]
        weights = np.array(saved["weights"], dtype=float)
        biases = np.array(saved["biases"], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{where}: missing or invalid activation, weights or biases")
    if activation != layer.activation:
        raise ValueError(f"{where}: activation '{activation}' doesn't match the config ('{layer.activation}')")
    if weights.shape != layer.weights.shape or biases.shape != layer.biases.shape:
        raise ValueError(f"{where}: expected weights {layer.weights.shape} and biases {layer.biases.shape}")
    if not (np.isfinite(weights).all() and np.isfinite(biases).all()):
        raise ValueError(f"{where}: weights and biases must be finite numbers")
    layer.weights = weights
    layer.biases = biases


def describe(record, epochs):
    text = f"epoch {record['epoch']}/{epochs}  loss {record['loss']:.4f}  accuracy {record['accuracy']:.2%}"
    if "val_loss" in record:
        text += f"  val_loss {record['val_loss']:.4f}  val_accuracy {record['val_accuracy']:.2%}"
    return text
