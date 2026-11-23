import json
import os
import tempfile
import unittest

import numpy as np

from src.nn.network import Network


def config(**changes):
    base = {
        "input_size": 6,
        "layers": [{"size": 5, "activation": "relu"}, {"size": 3, "activation": "softmax"}],
        "classes": ["Nothing", "Check", "Checkmate"],
        "optimizer": "adam",
        "learning_rate": 0.01,
        "epochs": 3,
        "seed": 0,
    }
    base.update(changes)
    return base


def data(n=60):
    rng = np.random.default_rng(1)
    return rng.normal(size=(n, 6)), rng.integers(0, 3, n)


class NetworkFile(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "my_torch_network_test.nn")
        self.x, self.y = data()
        self.net = Network(config())
        self.net.train(self.x, self.y)

    def tearDown(self):
        self.folder.cleanup()

    def saved(self):
        self.net.save(self.path)
        with open(self.path) as f:
            return json.load(f)

    def write(self, content):
        with open(self.path, "w") as f:
            json.dump(content, f)


class TestSaveLoad(NetworkFile):
    def test_same_predictions(self):
        self.net.save(self.path)
        loaded = Network.load(self.path)
        self.assertTrue(np.array_equal(loaded.predict(self.x), self.net.predict(self.x)))

    def test_file_content(self):
        content = self.saved()
        self.assertEqual(content["format"], "my_torch_network")
        self.assertEqual(content["version"], 1)
        self.assertEqual(content["config"], config())
        self.assertEqual(content["classes"], ["Nothing", "Check", "Checkmate"])
        self.assertEqual([len(layer["biases"]) for layer in content["layers"]], [5, 3])
        self.assertEqual(np.array(content["layers"][0]["weights"]).shape, (6, 5))

    def test_settings_are_kept(self):
        self.net.save(self.path)
        loaded = Network.load(self.path)
        self.assertEqual((loaded.learning_rate, loaded.epochs, loaded.loss), (0.01, 3, "cross_entropy"))
        self.assertEqual([layer.activation for layer in loaded.layers], ["relu", "softmax"])

    def test_training_continues(self):
        self.net.save(self.path)
        loaded = Network.load(self.path)
        before = loaded.evaluate(self.x, self.y)[0]
        loaded.train(self.x, self.y)
        self.assertLess(loaded.evaluate(self.x, self.y)[0], before)

    def test_save_overwrites(self):
        self.net.save(self.path)
        other = Network(config(seed=7))
        other.save(self.path)
        self.assertTrue(np.array_equal(Network.load(self.path).predict(self.x), other.predict(self.x)))

    def test_basic_config(self):
        net = Network.from_config("config/basic.conf")
        net.save(self.path)
        x = np.random.default_rng(2).integers(0, 2, (4, net.input_size))
        self.assertTrue(np.array_equal(Network.load(self.path).predict(x), net.predict(x)))


class TestLoadErrors(NetworkFile):
    def broken(self, change):
        content = self.saved()
        change(content)
        self.write(content)
        with self.assertRaises(ValueError):
            Network.load(self.path)

    def test_missing_file(self):
        with self.assertRaises(OSError):
            Network.load(os.path.join(self.folder.name, "nope.nn"))

    def test_not_json(self):
        with open(self.path, "w") as f:
            f.write("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")
        with self.assertRaises(ValueError):
            Network.load(self.path)

    def test_not_a_network(self):
        self.write([1, 2, 3])
        with self.assertRaises(ValueError):
            Network.load(self.path)
        self.write({"format": "something else"})
        with self.assertRaises(ValueError):
            Network.load(self.path)

    def test_wrong_version(self):
        self.broken(lambda content: content.update(version=2))

    def test_bad_config(self):
        self.broken(lambda content: content.pop("config"))
        self.broken(lambda content: content["config"].update(optimizer="nope"))

    def test_classes_mismatch(self):
        self.broken(lambda content: content.update(classes=["A", "B", "C"]))

    def test_missing_layer(self):
        self.broken(lambda content: content["layers"].pop())

    def test_wrong_shape(self):
        self.broken(lambda content: content["layers"][0]["weights"].pop())
        self.broken(lambda content: content["layers"][1]["biases"].append(0.0))

    def test_activation_mismatch(self):
        self.broken(lambda content: content["layers"][0].update(activation="tanh"))

    def test_missing_weights(self):
        self.broken(lambda content: content["layers"][0].pop("weights"))
        self.broken(lambda content: content["layers"].__setitem__(0, "weights"))

    def test_not_numbers(self):
        self.broken(lambda content: content["layers"][1]["biases"].__setitem__(0, "zero"))
        self.broken(lambda content: content["layers"][1]["biases"].__setitem__(0, float("nan")))


if __name__ == "__main__":
    unittest.main()
