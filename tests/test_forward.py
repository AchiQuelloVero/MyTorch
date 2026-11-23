import unittest

import numpy as np

from src.nn.activations import relu, sigmoid, softmax, tanh
from src.nn.network import Network

CONFIG = "config/basic.conf"


class TestActivations(unittest.TestCase):
    def test_values(self):
        z = np.array([[-2.0, 0.0, 3.0]])
        self.assertTrue(np.allclose(sigmoid(np.array([0.0])), 0.5))
        self.assertTrue(np.allclose(tanh(np.array([0.0])), 0.0))
        self.assertTrue(np.array_equal(relu(z), [[0.0, 0.0, 3.0]]))
        self.assertTrue(np.allclose(softmax(z).sum(axis=1), 1))

    def test_no_overflow(self):
        z = np.array([[-1000.0, 1000.0]])
        self.assertFalse(np.isnan(sigmoid(z)).any())
        self.assertFalse(np.isnan(softmax(z)).any())


class TestForward(unittest.TestCase):
    def setUp(self):
        self.net = Network.from_config(CONFIG)
        self.x = np.random.default_rng(0).integers(0, 2, (10, self.net.input_size))

    def test_shape(self):
        self.assertEqual(self.net.predict(self.x).shape, (10, 3))

    def test_probabilities(self):
        out = self.net.predict(self.x)
        self.assertTrue(np.allclose(out.sum(axis=1), 1))
        self.assertTrue((out >= 0).all())

    def test_single_sample(self):
        self.assertEqual(self.net.predict(self.x[0]).shape, (1, 3))

    def test_same_seed(self):
        other = Network.from_config(CONFIG)
        self.assertTrue(np.array_equal(self.net.predict(self.x), other.predict(self.x)))

    def test_wrong_input_size(self):
        with self.assertRaises(ValueError):
            self.net.predict(np.zeros((2, 10)))


class TestConfig(unittest.TestCase):
    def config(self, **changes):
        base = {
            "input_size": 4,
            "layers": [{"size": 3, "activation": "relu"}, {"size": 2, "activation": "softmax"}],
            "classes": ["A", "B"],
        }
        base.update(changes)
        return base

    def test_valid(self):
        self.assertEqual(Network(self.config()).predict(np.ones(4)).shape, (1, 2))

    def test_unknown_activation(self):
        with self.assertRaises(ValueError):
            Network(self.config(layers=[{"size": 2, "activation": "nope"}]))

    def test_classes_mismatch(self):
        with self.assertRaises(ValueError):
            Network(self.config(classes=["A", "B", "C"]))

    def test_missing_key(self):
        config = self.config()
        del config["layers"]
        with self.assertRaises(ValueError):
            Network(config)


if __name__ == "__main__":
    unittest.main()
