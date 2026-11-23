import unittest

import numpy as np

from src.nn.initializers import INITIALIZERS
from src.nn.network import Network


def config(**changes):
    base = {
        "input_size": 400,
        "layers": [
            {"size": 200, "activation": "relu"},
            {"size": 100, "activation": "relu"},
            {"size": 2, "activation": "softmax"},
        ],
        "classes": ["A", "B"],
        "seed": 0,
    }
    base.update(changes)
    return base


class TestInitializers(unittest.TestCase):
    def check_std(self, name, expected):
        weights = INITIALIZERS[name](np.random.default_rng(0), 800, 400)
        self.assertEqual(weights.shape, (800, 400))
        self.assertAlmostEqual(weights.mean(), 0, delta=0.01)
        self.assertAlmostEqual(weights.std(), expected, delta=expected * 0.02)

    def test_xavier(self):
        self.check_std("xavier", np.sqrt(2 / 1200))

    def test_he(self):
        self.check_std("he", np.sqrt(2 / 800))

    def test_lecun(self):
        self.check_std("lecun", np.sqrt(1 / 800))


class TestNetworkInit(unittest.TestCase):
    def test_init_from_config(self):
        net = Network(config(init="he"))
        self.assertEqual([layer.init for layer in net.layers], ["he", "he", "he"])

    def test_default_is_xavier(self):
        net = Network(config())
        self.assertEqual([layer.init for layer in net.layers], ["xavier", "xavier", "xavier"])

    def test_layer_overrides_config(self):
        layers = config()["layers"]
        layers[2] = {"size": 2, "activation": "softmax", "init": "xavier"}
        net = Network(config(init="he", layers=layers))
        self.assertEqual([layer.init for layer in net.layers], ["he", "he", "xavier"])

    def test_biases_start_at_zero(self):
        for layer in Network(config(init="lecun")).layers:
            self.assertFalse(layer.biases.any())

    def test_unknown_init(self):
        with self.assertRaises(ValueError):
            Network(config(init="nope"))

    def test_he_keeps_relu_signal(self):
        layers = [{"size": 256, "activation": "relu"} for _ in range(10)]
        layers.append({"size": 2, "activation": "softmax"})
        x = np.random.default_rng(1).normal(size=(64, 256))
        he = Network(config(input_size=256, layers=layers, init="he"))
        lecun = Network(config(input_size=256, layers=layers, init="lecun"))
        he.predict(x)
        lecun.predict(x)
        he_std = he.layers[-2].output.std()
        lecun_std = lecun.layers[-2].output.std()
        self.assertGreater(he_std, 0.3)
        self.assertGreater(he_std, lecun_std * 5)


if __name__ == "__main__":
    unittest.main()
