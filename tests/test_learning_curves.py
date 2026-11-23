import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    from scripts import learning_curves

BOARDS = [
    "8/1B4qK/5k2/5p2/8/8/8/4q3 w - - 1 51 Checkmate Black",
    "2k5/6R1/8/p1p2P2/2P1p3/1P2K1pP/P6r/8 b - - 1 46 Nothing",
    "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1 Check White",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing",
]
CONFIG = {
    "input_size": 776,
    "layers": [{"size": 8, "activation": "relu"}, {"size": 3, "activation": "softmax"}],
    "classes": ["Nothing", "Check", "Checkmate"],
    "optimizer": "adam",
    "learning_rate": 0.01,
    "batch_size": 2,
    "epochs": 3,
    "seed": 0,
}


def history(epochs, best):
    return [
        {
            "epoch": epoch,
            "loss": 1 / epoch,
            "accuracy": 1 - 0.5 / epoch,
            "val_loss": 0.5 + abs(epoch - best) / 10,
            "val_accuracy": 0.6 + epoch / 100,
        }
        for epoch in range(1, epochs + 1)
    ]


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
class TestLearningCurves(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        self.config = self.path / "tiny.conf"
        self.config.write_text(json.dumps(CONFIG))
        (self.path / "train.txt").write_text("\n".join(BOARDS) + "\n")
        (self.path / "val.txt").write_text("\n".join(BOARDS) + "\n")

    def tearDown(self):
        self.folder.cleanup()

    def assert_png(self, path):
        self.assertEqual(path.read_bytes()[:4], b"\x89PNG")

    def run_curves(self, boards=None, epochs=None):
        with contextlib.redirect_stderr(io.StringIO()):
            return learning_curves.curves(self.config, self.path, boards, epochs, self.path / "out")

    def test_best_record_has_the_lowest_val_loss(self):
        self.assertEqual(learning_curves.best_record(history(6, 4))["epoch"], 4)

    def test_train_val_graph(self):
        runs = [("first", history(4, 2)), ("second", history(9, 7))]
        learning_curves.plot_train_val(runs, 100, self.path / "graph.png")
        self.assert_png(self.path / "graph.png")

    def test_networks_of_different_lengths_on_the_same_graph(self):
        runs = [(f"net {number}", history(3 + number, 2)) for number in range(4)]
        learning_curves.plot_networks(runs, 100, self.path / "graph.png")
        self.assert_png(self.path / "graph.png")

    def test_writes_both_graphs(self):
        self.run_curves()
        self.assert_png(self.path / "out" / "curves_train_val.png")
        self.assert_png(self.path / "out" / "curves_networks.png")

    def test_one_curve_per_network(self):
        runs = self.run_curves()
        self.assertEqual([name for name, _ in runs], [run[0] for run in learning_curves.RUNS])
        self.assertTrue(all(len(run) == 3 for _, run in runs))

    def test_every_epoch_has_validation_results(self):
        for _, run in self.run_curves():
            self.assertTrue(all("val_loss" in record and "val_accuracy" in record for record in run))

    def test_epochs_override(self):
        self.assertTrue(all(len(run) == 2 for _, run in self.run_curves(epochs=2)))

    def test_missing_folder_exits_84(self):
        arguments = ["learning_curves", "--splits", str(self.path / "nope"), "--config", str(self.config)]
        err = io.StringIO()
        with mock.patch("sys.argv", arguments), contextlib.redirect_stderr(err):
            self.assertEqual(learning_curves.main(), 84)
        self.assertTrue(err.getvalue().startswith("learning_curves: "))


if __name__ == "__main__":
    unittest.main()
