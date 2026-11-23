import contextlib
import io
import unittest

import numpy as np

from src.nn.early_stopping import EarlyStopping
from src.nn.network import Network


class FakeLayer:
    def __init__(self):
        self.weights = np.zeros((2, 2))
        self.biases = np.zeros(2)


def config(**changes):
    base = {
        "input_size": 20,
        "layers": [{"size": 64, "activation": "relu"}, {"size": 2, "activation": "softmax"}],
        "classes": ["A", "B"],
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 16,
        "epochs": 60,
        "early_stopping": 3,
        "seed": 0,
    }
    base.update(changes)
    return base


def noise(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 20)), rng.integers(0, 2, n)


class TestEarlyStopping(unittest.TestCase):
    def answers(self, patience, losses):
        stopper = EarlyStopping(patience)
        layers = [FakeLayer()]
        return stopper, [stopper.should_stop(epoch, loss, layers) for epoch, loss in enumerate(losses, 1)]

    def test_stops_after_patience(self):
        stopper, answers = self.answers(2, [1.0, 0.8, 0.9, 0.85])
        self.assertEqual(answers, [False, False, False, True])
        self.assertEqual((stopper.best_epoch, stopper.best_loss), (2, 0.8))

    def test_improvement_resets_the_count(self):
        _, answers = self.answers(2, [1.0, 1.1, 0.9, 1.0, 1.0])
        self.assertEqual(answers, [False, False, False, False, True])

    def test_same_loss_is_not_an_improvement(self):
        _, answers = self.answers(1, [1.0, 1.0])
        self.assertEqual(answers, [False, True])

    def test_restores_a_copy_of_the_best(self):
        layer = FakeLayer()
        stopper = EarlyStopping(5)
        layer.weights += 1
        layer.biases += 2
        stopper.should_stop(1, 0.5, [layer])
        layer.weights += 10
        layer.biases += 10
        stopper.should_stop(2, 0.9, [layer])
        stopper.restore([layer])
        self.assertTrue((layer.weights == 1).all())
        self.assertTrue((layer.biases == 2).all())

    def test_nan_never_improves(self):
        layer = FakeLayer()
        layer.weights += 3
        stopper = EarlyStopping(2)
        answers = [stopper.should_stop(epoch, float("nan"), [layer]) for epoch in (1, 2)]
        stopper.restore([layer])
        self.assertEqual(answers, [False, True])
        self.assertTrue((layer.weights == 3).all())


class TestTrainingStops(unittest.TestCase):
    def setUp(self):
        self.x, self.y = noise(200, 1)
        self.x_val, self.y_val = noise(200, 2)

    def train(self, **changes):
        net = Network(config(**changes))
        return net, net.train(self.x, self.y, self.x_val, self.y_val)

    def best(self, history):
        return min(history, key=lambda record: record["val_loss"])

    def test_stops_when_overfitting(self):
        _, history = self.train()
        self.assertLess(len(history), 60)
        self.assertGreater(history[-1]["accuracy"], history[0]["accuracy"])
        self.assertGreater(history[-1]["val_loss"], self.best(history)["val_loss"])

    def test_patience_epochs_after_the_best(self):
        _, history = self.train(early_stopping=4)
        self.assertEqual(len(history), self.best(history)["epoch"] + 4)

    def test_keeps_the_best_network(self):
        net, history = self.train()
        self.assertAlmostEqual(net.evaluate(self.x_val, self.y_val)[0], self.best(history)["val_loss"])

    def test_keeps_the_best_even_without_stopping(self):
        net, history = self.train(epochs=15, early_stopping=100)
        self.assertEqual(len(history), 15)
        self.assertAlmostEqual(net.evaluate(self.x_val, self.y_val)[0], self.best(history)["val_loss"])

    def test_off(self):
        net, history = self.train(epochs=15, early_stopping=0)
        self.assertEqual(len(history), 15)
        self.assertAlmostEqual(net.evaluate(self.x_val, self.y_val)[0], history[-1]["val_loss"])

    def test_default_is_off(self):
        settings = config()
        del settings["early_stopping"]
        self.assertEqual(Network(settings).early_stopping, 0)

    def test_needs_validation_data(self):
        history = Network(config(epochs=15)).train(self.x, self.y)
        self.assertEqual(len(history), 15)

    def test_same_seed_same_stop(self):
        self.assertEqual(self.train()[1], self.train()[1])

    def test_negative_patience(self):
        with self.assertRaises(ValueError):
            Network(config(early_stopping=-1))

    def test_verbose_names_the_best_epoch(self):
        err = io.StringIO()
        net = Network(config())
        with contextlib.redirect_stderr(err):
            history = net.train(self.x, self.y, self.x_val, self.y_val, verbose=True)
        line = err.getvalue().strip().splitlines()[-1]
        self.assertTrue(line.startswith(f"best network: epoch {self.best(history)['epoch']} of {len(history)} "))


if __name__ == "__main__":
    unittest.main()
