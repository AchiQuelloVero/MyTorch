import tempfile
import unittest
from pathlib import Path

from scripts.evaluate import evaluate, report
from src.chess.encoding import INPUT_SIZE
from src.nn.network import Network

BOARDS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing",
]


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        net = Network({"input_size": INPUT_SIZE, "layers": [{"size": 3, "activation": "softmax"}],
                       "classes": ["Nothing", "Check", "Checkmate"]})
        net.layers[0].weights[:] = 0
        net.layers[0].biases[:] = [0, 1, 0]
        net.save(self.folder / "check.nn")
        (self.folder / "boards.txt").write_text("".join(f"{line}\n" for line in BOARDS))

    def test_report(self):
        text = report("command", *evaluate(self.folder / "check.nn", self.folder / "boards.txt"))
        self.assertIn("4 boards, accuracy **25.00%**, macro-F1 **0.1333**", text)
        self.assertIn("| Check | 1 | 25.00% | 100.00% | 0.4000 |", text)
        self.assertIn("| Nothing | 0 | 2 | 0 |", text)
        self.assertIn("| Check White | 0 | 1 | 0 | 0 | 0 |", text)
        self.assertIn("| 2-9 pieces | 1 | 0.00% |", text)
        self.assertNotIn("10-19 pieces", text)
        self.assertIn("| 20-32 pieces | 3 | 33.33% |", text)

    def test_missing_label(self):
        (self.folder / "boards.txt").write_text(f"{BOARDS[0]}\n{BOARDS[1].rsplit(' ', 1)[0]}\n")
        with self.assertRaisesRegex(ValueError, "boards.txt:2: missing label"):
            evaluate(self.folder / "check.nn", self.folder / "boards.txt")


if __name__ == "__main__":
    unittest.main()
