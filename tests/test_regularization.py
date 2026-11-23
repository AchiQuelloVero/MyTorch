import os
import tempfile
import unittest

import numpy as np

from src.nn.losses import cross_entropy
from src.nn.network import Network


def config(hidden="tanh", **changes):
    base = {
        "input_size": 3,
        "layers": [
            {"size": 4, "activation": hidden},
            {"size": 3, "activation": hidden},
            {"size": 2, "activation": "softmax"},
        ],
        "classes": ["A", "B"],
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 8,
        "epochs": 5,
        "seed": 1,
    }
    base.update(changes)
    return base


def big_config(**changes):
    base = {
        "input_size": 20,
        "layers": [{"size": 64, "activation": "relu"}, {"size": 2, "activation": "softmax"}],
        "classes": ["A", "B"],
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 16,
        "epochs": 40,
        "seed": 0,
    }
    base.update(changes)
    return base


def data(n=40, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    return x, np.eye(2)[(x[:, 0] > 0).astype(int)]


def noise(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 20)), rng.integers(0, 2, n)


def numerical_gradient(cost, values, eps=1e-5):
    grad = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        old = values[index]
        values[index] = old + eps
        plus = cost()
        values[index] = old - eps
        minus = cost()
        values[index] = old
        grad[index] = (plus - minus) / (2 * eps)
    return grad


class Regularization(unittest.TestCase):
    def setUp(self):
        self.x, self.y = data()

    def check_gradients(self, net, cost):
        for layer in net.layers:
            self.assertTrue(np.allclose(layer.grad_weights, numerical_gradient(cost, layer.weights), atol=1e-6))
            self.assertTrue(np.allclose(layer.grad_biases, numerical_gradient(cost, layer.biases), atol=1e-6))

    def check_dropout_gradients(self, net):
        def cost():
            net.rng = np.random.default_rng(5)
            return cross_entropy(net.forward(self.x, True), self.y) + net.penalty()
        cost()
        net.backward(self.y)
        self.check_gradients(net, cost)


class TestL2(Regularization):
    def test_gradient_check(self):
        net = Network(config(l2=0.1))
        net.predict(self.x)
        net.backward(self.y)
        self.check_gradients(net, lambda: net.cost(self.x, self.y))

    def test_penalty_value(self):
        net = Network(config(l2=0.5))
        for layer in net.layers:
            layer.weights[:] = 2.0
        self.assertAlmostEqual(net.penalty(), 30)

    def test_biases_are_not_penalized(self):
        plain, penalized = Network(config()), Network(config(l2=0.1))
        for net in (plain, penalized):
            for layer in net.layers:
                layer.biases += 0.5
            net.predict(self.x)
            net.backward(self.y)
        for one, other in zip(plain.layers, penalized.layers):
            self.assertTrue(np.array_equal(one.grad_biases, other.grad_biases))
            self.assertFalse(np.array_equal(one.grad_weights, other.grad_weights))

    def test_shrinks_the_weights(self):
        def size(net):
            net.train(self.x, self.y)
            return sum(np.sum(layer.weights ** 2) for layer in net.layers)
        self.assertLess(size(Network(config(l2=0.1, epochs=30))), size(Network(config(epochs=30))))

    def test_cost_has_the_penalty(self):
        net = Network(config(l2=0.1))
        self.assertAlmostEqual(net.cost(self.x, self.y) - net.evaluate(self.x, self.y)[0], net.penalty())

    def test_reported_loss_has_no_penalty(self):
        plain, penalized = Network(config()), Network(config(l2=10.0))
        self.assertAlmostEqual(plain.train_step(self.x, self.y), penalized.train_step(self.x, self.y))

    def test_zero_changes_nothing(self):
        plain, zero = Network(config()), Network(config(l2=0))
        self.assertEqual(plain.train(self.x, self.y), zero.train(self.x, self.y))

    def test_default_is_off(self):
        self.assertEqual(Network(config()).l2, 0)

    def test_negative(self):
        with self.assertRaises(ValueError):
            Network(config(l2=-0.1))

    def test_not_a_number(self):
        with self.assertRaises(ValueError):
            Network(config(l2="much"))


class TestDropout(Regularization):
    def wide(self, dropout):
        layers = [{"size": 200, "activation": "relu"}, {"size": 2, "activation": "softmax"}]
        net = Network(config(layers=layers, dropout=dropout))
        net.forward(np.ones((1000, 3)), True)
        return net

    def test_gradient_check_tanh(self):
        self.check_dropout_gradients(Network(config("tanh", dropout=0.4)))

    def test_gradient_check_relu(self):
        net = Network(config("relu", dropout=0.4))
        for layer in net.layers:
            layer.biases += 0.1
        self.check_dropout_gradients(net)

    def test_gradient_check_with_l2(self):
        self.check_dropout_gradients(Network(config("tanh", dropout=0.4, l2=0.1)))

    def test_drop_rate(self):
        self.assertAlmostEqual(np.mean(self.wide(0.3).layers[0].mask == 0), 0.3, delta=0.01)

    def test_kept_neurons_are_scaled(self):
        mask = self.wide(0.3).layers[0].mask
        self.assertTrue(np.allclose(np.unique(mask), [0, 1 / 0.7]))
        self.assertAlmostEqual(np.mean(mask), 1, delta=0.02)

    def test_output_layer_is_never_dropped(self):
        net = Network(config(dropout=0.9))
        output = net.forward(self.x, True)
        self.assertEqual(net.layers[-1].mask, 1.0)
        self.assertTrue(np.allclose(output.sum(axis=1), 1))

    def test_training_forward_is_random(self):
        net = Network(config(dropout=0.5))
        self.assertFalse(np.array_equal(net.forward(self.x, True), net.forward(self.x, True)))

    def test_predict_uses_every_neuron(self):
        plain, dropped = Network(config()), Network(config(dropout=0.5))
        self.assertTrue(np.array_equal(dropped.predict(self.x), plain.predict(self.x)))
        self.assertTrue(np.array_equal(dropped.predict(self.x), dropped.predict(self.x)))

    def test_evaluate_is_stable_after_training(self):
        net = Network(config(dropout=0.5))
        net.train(self.x, self.y)
        self.assertEqual(net.evaluate(self.x, self.y), net.evaluate(self.x, self.y))

    def test_zero_changes_nothing(self):
        plain, zero = Network(config()), Network(config(dropout=0))
        self.assertEqual(plain.train(self.x, self.y), zero.train(self.x, self.y))

    def test_same_seed_same_training(self):
        one, other = Network(config(dropout=0.5)), Network(config(dropout=0.5))
        self.assertEqual(one.train(self.x, self.y), other.train(self.x, self.y))
        self.assertTrue(np.array_equal(one.predict(self.x), other.predict(self.x)))

    def test_changes_the_training(self):
        plain, dropped = Network(config()), Network(config(dropout=0.5))
        self.assertNotEqual(plain.train(self.x, self.y), dropped.train(self.x, self.y))

    def test_still_learns(self):
        x, y = data(300, 3)
        x_val, y_val = data(300, 4)
        layers = [{"size": 32, "activation": "relu"}, {"size": 2, "activation": "softmax"}]
        net = Network(config(layers=layers, dropout=0.2, epochs=30))
        net.train(x, y)
        self.assertGreater(net.evaluate(x_val, y_val)[1], 0.9)

    def test_default_is_off(self):
        self.assertEqual(Network(config()).dropout, 0)

    def test_out_of_range(self):
        for value in (-0.1, 1, 1.5):
            with self.assertRaises(ValueError):
                Network(config(dropout=value))


class TestOverfitting(unittest.TestCase):
    def setUp(self):
        self.x, self.y = noise(200, 1)
        self.x_val, self.y_val = noise(200, 2)

    def val_loss(self, **changes):
        net = Network(big_config(**changes))
        net.train(self.x, self.y)
        return net.evaluate(self.x_val, self.y_val)[0]

    def test_l2_helps(self):
        self.assertLess(self.val_loss(l2=0.01), self.val_loss())

    def test_dropout_helps(self):
        self.assertLess(self.val_loss(dropout=0.5), self.val_loss())


class TestSavedNetwork(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "my_torch_network_test.nn")
        self.x, self.y = data()
        self.net = Network(config(l2=0.01, dropout=0.3))
        self.net.train(self.x, self.y)
        self.net.save(self.path)
        self.loaded = Network.load(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def test_settings_are_kept(self):
        self.assertEqual((self.loaded.l2, self.loaded.dropout), (0.01, 0.3))

    def test_same_predictions(self):
        self.assertTrue(np.array_equal(self.loaded.predict(self.x), self.net.predict(self.x)))

    def test_training_continues(self):
        before = self.loaded.predict(self.x)
        self.loaded.train(self.x, self.y)
        self.assertFalse(np.array_equal(self.loaded.predict(self.x), before))


if __name__ == "__main__":
    unittest.main()
