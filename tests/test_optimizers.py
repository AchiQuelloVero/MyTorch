import unittest

import numpy as np

from src.nn.network import Network
from src.nn.optimizers import SGD, Adam, Momentum


class FakeLayer:
    def __init__(self, grad):
        self.weights = np.zeros(3)
        self.biases = np.zeros(1)
        self.grad_weights = np.array(grad, dtype=float)
        self.grad_biases = np.zeros(1)


def config(**changes):
    base = {
        "input_size": 10,
        "layers": [{"size": 16, "activation": "tanh"}, {"size": 2, "activation": "softmax"}],
        "classes": ["A", "B"],
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "batch_size": 16,
        "epochs": 15,
        "seed": 0,
    }
    base.update(changes)
    return base


def problem(n=400):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(n, 10))
    y = (x[:, 0] + x[:, 1] - x[:, 2] > 0).astype(int)
    return x, y


class TestOptimizerSteps(unittest.TestCase):
    def test_sgd(self):
        layer = FakeLayer([1.0, -2.0, 0.0])
        SGD(0.1).step([layer])
        self.assertTrue(np.allclose(layer.weights, [-0.1, 0.2, 0.0]))

    def test_momentum_speeds_up(self):
        layer = FakeLayer([1.0, 1.0, 1.0])
        optimizer = Momentum(0.1, 0.9)
        optimizer.step([layer])
        first = layer.weights.copy()
        optimizer.step([layer])
        self.assertTrue(np.allclose(first, -0.1))
        self.assertTrue(np.allclose(layer.weights - first, -0.19))

    def test_adam_first_step_is_the_learning_rate(self):
        layer = FakeLayer([0.001, -50.0, 3.0])
        optimizer = Adam(0.01)
        optimizer.step([layer])
        self.assertTrue(np.allclose(layer.weights, [-0.01, 0.01, -0.01], rtol=1e-3))
        self.assertEqual(optimizer.t, 1)


class TestTraining(unittest.TestCase):
    def final_loss(self, **changes):
        x, y = problem()
        return Network(config(**changes)).train(x, y)[-1]["loss"]

    def test_loss_goes_down(self):
        x, y = problem()
        history = Network(config(optimizer="adam")).train(x, y)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(history[-1]["accuracy"], 0.9)

    def test_momentum_and_adam_beat_sgd(self):
        sgd = self.final_loss(optimizer="sgd")
        self.assertLess(self.final_loss(optimizer="momentum"), sgd)
        self.assertLess(self.final_loss(optimizer="adam"), sgd)

    def test_steps_per_epoch(self):
        x, y = problem(50)
        for batch_size, steps in ((1, 50), (16, 4), (50, 1), (500, 1)):
            net = Network(config(optimizer="adam", batch_size=batch_size, epochs=2))
            net.train(x, y)
            self.assertEqual(net.optimizer.t, steps * 2)

    def test_learning_rate_decay(self):
        x, y = problem(50)
        net = Network(config(learning_rate=0.3, lr_decay=1.0, epochs=3))
        net.train(x, y)
        self.assertAlmostEqual(net.optimizer.learning_rate, 0.1)

    def test_history(self):
        x, y = problem()
        history = Network(config(epochs=3)).train(x[:300], y[:300], x[300:], y[300:])
        self.assertEqual([record["epoch"] for record in history], [1, 2, 3])
        for record in history:
            self.assertEqual(set(record), {"epoch", "loss", "accuracy", "val_loss", "val_accuracy"})

    def test_same_seed_same_training(self):
        x, y = problem()
        first = Network(config()).train(x, y)
        second = Network(config()).train(x, y)
        self.assertEqual(first, second)

    def test_indices_or_one_hot(self):
        x, y = problem()
        indices = Network(config(epochs=2)).train(x, y)
        one_hot = Network(config(epochs=2)).train(x, np.eye(2)[y])
        self.assertEqual(indices, one_hot)

    def test_uint8_inputs(self):
        x, y = problem()
        history = Network(config(epochs=2)).train((x > 0).astype(np.uint8), y)
        self.assertEqual(len(history), 2)

    def test_class_weights(self):
        x, y = problem()
        y[:300] = 0
        plain = Network(config(optimizer="adam")).train(x, y)
        weighted = Network(config(optimizer="adam")).train(x, y, class_weights=[1.0, 1.0])
        self.assertEqual(plain, weighted)
        net = Network(config(optimizer="adam"))
        net.train(x, y, class_weights=[0.1, 10.0])
        self.assertGreater(net.predict(x).argmax(axis=1).mean(), y.mean())

    def test_wrong_sizes(self):
        x, y = problem()
        net = Network(config())
        with self.assertRaises(ValueError):
            net.train(x, y[:10])
        with self.assertRaises(ValueError):
            net.train(x, y, class_weights=[1.0])
        with self.assertRaises(ValueError):
            net.train(x, y + 5)


class TestTrainingConfig(unittest.TestCase):
    def test_bad_values(self):
        for changes in ({"optimizer": "nope"}, {"learning_rate": 0}, {"lr_decay": -1},
                        {"batch_size": 0}, {"epochs": 0}):
            with self.assertRaises(ValueError):
                Network(config(**changes))

    def test_defaults(self):
        net = Network({"input_size": 2, "layers": [{"size": 2, "activation": "softmax"}], "classes": ["A", "B"]})
        self.assertIsInstance(net.optimizer, SGD)
        self.assertEqual((net.learning_rate, net.batch_size, net.epochs, net.lr_decay), (0.01, 32, 10, 0.0))

    def test_adam_and_momentum_settings(self):
        self.assertEqual(Network(config(optimizer="momentum", momentum=0.5)).optimizer.beta, 0.5)
        adam = Network(config(optimizer="adam", beta1=0.8, beta2=0.99)).optimizer
        self.assertEqual((adam.beta1, adam.beta2), (0.8, 0.99))


if __name__ == "__main__":
    unittest.main()
