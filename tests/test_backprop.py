import unittest

import numpy as np

from src.nn.losses import binary_cross_entropy, cross_entropy, mse
from src.nn.network import Network

XOR_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_OUTPUTS = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)


def small_network(hidden, output, loss):
    return Network({
        "input_size": 3,
        "layers": [
            {"size": 4, "activation": hidden},
            {"size": 3, "activation": hidden},
            {"size": 2, "activation": output},
        ],
        "classes": ["A", "B"],
        "loss": loss,
        "learning_rate": 0.5,
        "seed": 1,
    })


def numerical_gradient(net, x, y, values, weights=None, eps=1e-5):
    grad = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        old = values[index]
        values[index] = old + eps
        plus = net.cost(x, y, weights)
        values[index] = old - eps
        minus = net.cost(x, y, weights)
        values[index] = old
        grad[index] = (plus - minus) / (2 * eps)
    return grad


class TestLosses(unittest.TestCase):
    def test_weights_of_one_change_nothing(self):
        y = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        out = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
        ones = np.ones(3)
        self.assertAlmostEqual(cross_entropy(out, y, ones), cross_entropy(out, y))
        self.assertAlmostEqual(mse(out, y, ones), mse(out, y))

    def test_weights_favor_a_sample(self):
        y = np.array([[0.0, 1.0], [1.0, 0.0]])
        out = np.array([[0.1, 0.9], [0.1, 0.9]])
        self.assertGreater(cross_entropy(out, y, np.array([1.0, 5.0])), cross_entropy(out, y))

    def test_perfect_prediction(self):
        y = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(cross_entropy(y, y), 0)
        self.assertAlmostEqual(binary_cross_entropy(y, y), 0)
        self.assertAlmostEqual(mse(y, y), 0)

    def test_wrong_prediction_costs_more(self):
        y = np.array([[0.0, 1.0]])
        good = np.array([[0.1, 0.9]])
        bad = np.array([[0.9, 0.1]])
        self.assertLess(cross_entropy(good, y), cross_entropy(bad, y))
        self.assertLess(mse(good, y), mse(bad, y))

    def test_no_log_of_zero(self):
        y = np.array([[0.0, 1.0]])
        self.assertTrue(np.isfinite(cross_entropy(np.array([[1.0, 0.0]]), y)))
        self.assertTrue(np.isfinite(binary_cross_entropy(np.array([[1.0, 0.0]]), y)))


class TestGradientCheck(unittest.TestCase):
    def check(self, hidden, output, loss, weights=None):
        net = small_network(hidden, output, loss)
        rng = np.random.default_rng(2)
        x = rng.normal(size=(5, 3))
        y = np.eye(2)[rng.integers(0, 2, 5)]
        net.predict(x)
        net.backward(y, weights)
        for layer in net.layers:
            expected = numerical_gradient(net, x, y, layer.weights, weights)
            self.assertTrue(np.allclose(layer.grad_weights, expected, atol=1e-6))
            expected = numerical_gradient(net, x, y, layer.biases, weights)
            self.assertTrue(np.allclose(layer.grad_biases, expected, atol=1e-6))

    def test_softmax_cross_entropy(self):
        self.check("tanh", "softmax", "cross_entropy")

    def test_sigmoid_binary_cross_entropy(self):
        self.check("sigmoid", "sigmoid", "binary_cross_entropy")

    def test_mse(self):
        self.check("tanh", "sigmoid", "mse")

    def test_relu(self):
        self.check("relu", "softmax", "cross_entropy")

    def test_sample_weights(self):
        self.check("tanh", "softmax", "cross_entropy", np.array([1.0, 3.0, 0.5, 2.0, 1.0]))
        self.check("tanh", "sigmoid", "mse", np.array([0.2, 1.0, 1.0, 4.0, 1.0]))


class TestLearning(unittest.TestCase):
    def test_cost_goes_down(self):
        net = small_network("tanh", "softmax", "cross_entropy")
        x = np.random.default_rng(3).normal(size=(20, 3))
        y = np.eye(2)[(x[:, 0] > 0).astype(int)]
        first = net.train_step(x, y)
        for _ in range(50):
            last = net.train_step(x, y)
        self.assertLess(last, first)

    def test_xor(self):
        net = Network({
            "input_size": 2,
            "layers": [{"size": 4, "activation": "tanh"}, {"size": 2, "activation": "softmax"}],
            "classes": ["0", "1"],
            "learning_rate": 0.5,
            "seed": 0,
        })
        for _ in range(3000):
            net.train_step(XOR_INPUTS, XOR_OUTPUTS)
        predicted = net.predict(XOR_INPUTS).argmax(axis=1)
        self.assertTrue(np.array_equal(predicted, XOR_OUTPUTS.argmax(axis=1)))


class TestConfigChecks(unittest.TestCase):
    def test_softmax_hidden_layer(self):
        with self.assertRaises(ValueError):
            small_network("softmax", "softmax", "cross_entropy")

    def test_loss_and_output_mismatch(self):
        with self.assertRaises(ValueError):
            small_network("tanh", "sigmoid", "cross_entropy")

    def test_unknown_loss(self):
        with self.assertRaises(ValueError):
            small_network("tanh", "softmax", "nope")


if __name__ == "__main__":
    unittest.main()
