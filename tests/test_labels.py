import glob
import unittest

from src.chess.labels import LabelError, decode_labels, encode_labels

DATASET = glob.glob("dataset/*/*.txt")

PDF_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]
PDF_BASIC = ["Checkmate", "Nothing", "Nothing", "Check", "Nothing"]
PDF_FULL = ["Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing"]
PDF_SIDES = [fen.split()[1] for fen in PDF_FENS]

INVALID = ["", "check", "Mate", "Nothing White", "Check Red", "Check White now"]


class TestEncodeLabels(unittest.TestCase):
    def test_basic_and_full_labels(self):
        labels = ["Nothing", "Check", "Checkmate", "Check White", "Checkmate Black"]
        self.assertEqual(encode_labels(labels).tolist(), [0, 1, 2, 1, 2])

    def test_classes_order(self):
        classes = ("Checkmate", "Nothing", "Check")
        self.assertEqual(encode_labels(["Nothing", "Check White"], classes).tolist(), [1, 2])

    def test_invalid_labels(self):
        for label in INVALID:
            with self.subTest(label), self.assertRaisesRegex(LabelError, "^board 2: "):
                encode_labels(["Check", label])


class TestDecodeLabels(unittest.TestCase):
    def test_pdf_example(self):
        indices = encode_labels(PDF_BASIC)
        self.assertEqual(decode_labels(indices, PDF_SIDES), PDF_BASIC)
        self.assertEqual(decode_labels(indices, PDF_SIDES, full=True), PDF_FULL)

    def test_sides_must_match(self):
        with self.assertRaises(ValueError):
            decode_labels([0, 1], ["w"], full=True)

    @unittest.skipUnless(DATASET, "dataset/ not found")
    def test_dataset_round_trip(self):
        lines = []
        for path in DATASET:
            with open(path) as f:
                lines.extend(line.split() for line in f)
        labels = [" ".join(words[6:]) for words in lines]
        sides = [words[1] for words in lines]
        self.assertEqual(decode_labels(encode_labels(labels), sides, full=True), labels)


if __name__ == "__main__":
    unittest.main()
