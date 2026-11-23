import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_dataset import build
from src.chess.dataset import ChessFileError, class_weights, deduplicate, read_chessfile, split_indices

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
AFTER_E5 = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1"
KINGS = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


class TestReadChessfile(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write(self, text):
        path = Path(self.folder.name) / "boards.txt"
        path.write_text(text)
        return path

    def test_fens_and_labels(self):
        path = self.write(f"{START} Nothing\n{AFTER_E4} Check White\n{KINGS}\n\n\n")
        self.assertEqual(read_chessfile(path), ([START, AFTER_E4, KINGS], ["Nothing", "Check White", ""]))

    def test_errors(self):
        cases = {
            "blank line in the middle": (f"{START}\n\n{KINGS}\n", ":2: "),
            "FEN with 4 fields": ("8/8/8/8/8/8/8/k1K5 w - -\n", ":1: "),
            "empty file": ("\n  \n", "no chessboard"),
        }
        for reason, (text, message) in cases.items():
            with self.subTest(reason), self.assertRaisesRegex(ChessFileError, message):
                read_chessfile(self.write(text))


class TestDeduplicate(unittest.TestCase):
    def test_duplicates_are_removed(self):
        fens = [START, START.replace("0 1", "4 9"), START.replace("KQkq", "-"), AFTER_E4, AFTER_E5, KINGS]
        labels = ["a", "b", "c", "d", "e", "f"]
        self.assertEqual(deduplicate(fens, labels), ([START, AFTER_E4, KINGS], ["a", "d", "f"]))


class TestSplitIndices(unittest.TestCase):
    labels = ["Nothing"] * 500 + ["Check"] * 300 + ["Checkmate"] * 200

    def test_stratified(self):
        parts = split_indices(self.labels, (0.8, 0.1, 0.1), seed=1)
        self.assertEqual([len(part) for part in parts], [800, 100, 100])
        for part, fraction in zip(parts, (0.8, 0.1, 0.1)):
            counts = {label: sum(self.labels[i] == label for i in part) for label in set(self.labels)}
            self.assertEqual(counts, {"Nothing": 500 * fraction, "Check": 300 * fraction, "Checkmate": 200 * fraction})

    def test_every_line_exactly_once(self):
        parts = split_indices(self.labels, (0.8, 0.1, 0.1), seed=1)
        self.assertEqual(sorted(np.concatenate(parts).tolist()), list(range(len(self.labels))))

    def test_seed(self):
        first, again, other = (split_indices(self.labels, (0.9, 0.1), seed) for seed in (1, 1, 2))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, again)))
        self.assertFalse(np.array_equal(first[1], other[1]))

    def test_invalid_fractions(self):
        for fractions in [(0.8, 0.1), (0.9, 0.2, -0.1), (1.0, 0.0)]:
            with self.subTest(fractions), self.assertRaises(ValueError):
                split_indices(self.labels, fractions, seed=1)


class TestClassWeights(unittest.TestCase):
    y = np.array([0] * 500 + [1] * 300 + [2] * 200)

    def test_inverse_frequency(self):
        weights = class_weights(self.y, 3)
        np.testing.assert_allclose(weights, [1000 / 1500, 1000 / 900, 1000 / 600])
        np.testing.assert_allclose(weights * np.bincount(self.y), [1000 / 3] * 3)
        self.assertAlmostEqual(weights[self.y].mean(), 1)

    def test_balanced_classes_weigh_one(self):
        np.testing.assert_allclose(class_weights(np.array([0, 1, 2, 2, 1, 0]), 3), [1, 1, 1])

    def test_errors(self):
        for reason, y in {"missing class": [0, 0, 1], "label out of range": [0, 1, 2, 3]}.items():
            with self.subTest(reason), self.assertRaisesRegex(ValueError, "every class must appear"):
                class_weights(np.array(y), 3)


class TestBuildDataset(unittest.TestCase):
    def test_same_seed_same_files(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder)
            (dataset / "boards").mkdir()
            lines = [f"{fen} Nothing\n" for fen in (START, AFTER_E4, KINGS)]
            (dataset / "boards" / "a.txt").write_text("".join(lines * 10))
            outputs = []
            for _ in range(2):
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    build(dataset, dataset / "splits", seed=1)
                self.assertIn("30 boards in 1 files", printed.getvalue())
                outputs.append([(dataset / "splits" / f"{split}.txt").read_bytes() for split in ("train", "val", "test")])
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(sum(text.count(b"\n") for text in outputs[1]), 3)


if __name__ == "__main__":
    unittest.main()
