import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

from src.generator import USAGE, UsageError, main, parse_args
from src.nn.network import Network

CONFIG = {
    "input_size": 4,
    "layers": [{"size": 3, "activation": "relu"}, {"size": 2, "activation": "softmax"}],
    "classes": ["A", "B"],
}


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class TestParseArgs(unittest.TestCase):
    def test_pairs(self):
        self.assertEqual(parse_args(["a.conf", "3", "b.conf", "1"]), [("a.conf", 3), ("b.conf", 1)])

    def test_errors(self):
        for args in ([], ["a.conf"], ["a.conf", "2", "b.conf"], ["a.conf", "0"], ["a.conf", "-1"],
                     ["a.conf", "two"], ["a.conf", "1.5"]):
            with self.subTest(args=args), self.assertRaises(UsageError):
                parse_args(args)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.folder = tempfile.TemporaryDirectory()
        os.chdir(self.folder.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.folder.cleanup()

    def write_config(self, name, config):
        with open(name, "w") as f:
            json.dump(config, f)

    def seed_of(self, path):
        with open(path) as f:
            return json.load(f)["config"]["seed"]

    def test_help(self):
        code, out, _ = run(["--help"])
        self.assertEqual((code, out.strip()), (0, USAGE))

    def test_generates_the_files(self):
        self.write_config("basic_network.conf", CONFIG)
        code, out, err = run(["basic_network.conf", "3"])
        self.assertEqual((code, out, err), (0, "", ""))
        self.assertEqual(sorted(name for name in os.listdir() if name.endswith(".nn")),
                         ["basic_network_1.nn", "basic_network_2.nn", "basic_network_3.nn"])

    def test_networks_are_different(self):
        self.write_config("net.conf", CONFIG)
        run(["net.conf", "2"])
        first = Network.load("net_1.nn").layers[0].weights
        second = Network.load("net_2.nn").layers[0].weights
        self.assertFalse(np.array_equal(first, second))

    def test_seeds_follow_the_config(self):
        self.write_config("net.conf", {**CONFIG, "seed": 10})
        run(["net.conf", "3"])
        self.assertEqual([self.seed_of(f"net_{i}.nn") for i in (1, 2, 3)], [10, 11, 12])
        expected = Network({**CONFIG, "seed": 11}).layers[0].weights
        self.assertTrue(np.array_equal(Network.load("net_2.nn").layers[0].weights, expected))

    def test_random_seed_is_saved(self):
        self.write_config("net.conf", CONFIG)
        run(["net.conf", "1"])
        seed = self.seed_of("net_1.nn")
        self.assertIsInstance(seed, int)
        expected = Network({**CONFIG, "seed": seed}).layers[0].weights
        self.assertTrue(np.array_equal(Network.load("net_1.nn").layers[0].weights, expected))

    def test_several_configs(self):
        self.write_config("a.conf", CONFIG)
        os.mkdir("config")
        self.write_config("config/b.conf", CONFIG)
        code, _, _ = run(["a.conf", "2", "config/b.conf", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(name for name in os.listdir() if name.endswith(".nn")),
                         ["a_1.nn", "a_2.nn", "b_1.nn"])

    def test_errors_write_nothing(self):
        self.write_config("good.conf", CONFIG)
        self.write_config("bad.conf", {**CONFIG, "classes": ["A"]})
        self.write_config("list.conf", [1, 2])
        self.write_config("seed.conf", {**CONFIG, "seed": "abc"})
        with open("text.conf", "w") as f:
            f.write("layers = 3\n")
        for args in (["good.conf"], ["good.conf", "0"], ["missing.conf", "1"], ["text.conf", "1"],
                     ["list.conf", "1"], ["seed.conf", "1"], ["good.conf", "2", "bad.conf", "1"]):
            with self.subTest(args=args):
                code, out, err = run(args)
                self.assertEqual(code, 84)
                self.assertEqual(out, "")
                self.assertTrue(err.startswith("my_torch_generator: "))
                self.assertEqual([name for name in os.listdir() if name.endswith(".nn")], [])


if __name__ == "__main__":
    unittest.main()
