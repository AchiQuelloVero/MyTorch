import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.analyzer import USAGE, Arguments, UsageError, main, parse_args
from src.chess.encoding import INPUT_SIZE
from src.nn.network import Network
from tests.test_fen import INVALID as INVALID_FENS
from tests.test_labels import INVALID as INVALID_LABELS

BOARDS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
]
LABELS = ["Checkmate Black", "Nothing", "Nothing", "Check White"]


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class TestParseArgs(unittest.TestCase):
    def test_valid(self):
        cases = {
            "predict": (["--predict", "net.nn", "boards.txt"], Arguments("predict", "net.nn", "boards.txt", "net.nn")),
            "train": (["--train", "net.nn", "boards.txt"], Arguments("train", "net.nn", "boards.txt", "net.nn")),
            "train and save": (["--train", "--save", "new.nn", "net.nn", "boards.txt"],
                               Arguments("train", "net.nn", "boards.txt", "new.nn")),
            "any order": (["net.nn", "--save", "new.nn", "boards.txt", "--train"],
                          Arguments("train", "net.nn", "boards.txt", "new.nn")),
        }
        for reason, (args, expected) in cases.items():
            with self.subTest(reason):
                self.assertEqual(parse_args(args), expected)

    def test_errors(self):
        cases = {
            "no mode": ["net.nn", "boards.txt"],
            "both modes": ["--predict", "--train", "net.nn", "boards.txt"],
            "mode twice": ["--train", "--train", "net.nn", "boards.txt"],
            "save without train": ["--predict", "--save", "new.nn", "net.nn", "boards.txt"],
            "save without a file": ["--train", "net.nn", "boards.txt", "--save"],
            "save twice": ["--train", "--save", "a.nn", "--save", "b.nn", "net.nn", "boards.txt"],
            "missing file": ["--predict", "net.nn"],
            "extra file": ["--predict", "net.nn", "boards.txt", "more.txt"],
            "unknown option": ["--predict", "--verbose", "net.nn", "boards.txt"],
            "help with other arguments": ["--help", "--predict", "net.nn", "boards.txt"],
        }
        for reason, args in cases.items():
            with self.subTest(reason), self.assertRaises(UsageError):
                parse_args(args)


class TestMain(unittest.TestCase):
    def test_help(self):
        for flag in ("--help", "-h"):
            with self.subTest(flag):
                self.assertEqual(run([flag]), (0, USAGE + "\n", ""))

    def test_usage_error(self):
        code, out, err = run(["--predict", "net.nn"])
        self.assertEqual((code, out), (84, ""))
        self.assertIn("expected LOADFILE and CHESSFILE", err)


class TestPredict(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        for output in ("basic", "full"):
            net = Network({"input_size": INPUT_SIZE, "layers": [{"size": 3, "activation": "softmax"}],
                           "classes": ["Nothing", "Check", "Checkmate"], "output": output})
            net.layers[0].weights[:] = 0
            net.layers[0].biases[:] = [0, 1, 0]
            net.save(self.folder / f"{output}.nn")

    def predict(self, lines, network="basic.nn"):
        path = self.folder / "boards.txt"
        path.write_text("".join(f"{line}\n" for line in lines))
        return run(["--predict", str(self.folder / network), str(path)])

    def test_one_line_per_board(self):
        self.assertEqual(self.predict(BOARDS), (0, "Check\n" * 4, ""))

    def test_full_output_adds_the_attacker(self):
        expected = "Check Black\nCheck Black\nCheck White\nCheck White\n"
        self.assertEqual(self.predict(BOARDS, "full.nn"), (0, expected, ""))

    def test_accuracy_on_stderr(self):
        labels = ["Checkmate Black", "Nothing", "Check White", "Check"]
        code, out, err = self.predict([f"{board} {label}" for board, label in zip(BOARDS, labels)])
        self.assertEqual((code, out, err), (0, "Check\n" * 4, "accuracy 50.00% on 4 boards\n"))

    def test_no_accuracy_when_a_label_is_missing(self):
        self.assertEqual(self.predict([f"{BOARDS[0]} Check"] + BOARDS[1:]), (0, "Check\n" * 4, ""))

    def test_errors(self):
        cases = {
            "missing LOADFILE": (BOARDS, "missing.nn", "missing.nn"),
            "not a network": (BOARDS, "boards.txt", "not a JSON file"),
        }
        for reason, (lines, network, message) in cases.items():
            with self.subTest(reason):
                code, out, err = self.predict(lines, network)
                self.assertEqual((code, out), (84, ""))
                self.assertIn(message, err)

    def test_corrupted_networks(self):
        def corrupt(change):
            data = json.loads((self.folder / "basic.nn").read_text())
            change(data)
            return json.dumps(data)

        other_input = Network({"input_size": INPUT_SIZE - 3, "layers": [{"size": 3, "activation": "softmax"}],
                               "classes": ["Nothing", "Check", "Checkmate"]})
        other_input.save(self.folder / "other_input.nn")
        cases = {
            "invalid JSON": ('{"format": ', "not a JSON file"),
            "not a JSON object": ("[]", "not a my_torch_network file"),
            "wrong version": (corrupt(lambda d: d.update(version=99)), "unsupported version 99"),
            "missing config": (corrupt(lambda d: d.pop("config")), "config"),
            "invalid config": (corrupt(lambda d: d["config"].pop("layers")), "invalid config"),
            "missing layers": (corrupt(lambda d: d.pop("layers")), "expected 1 layers"),
            "wrong weights shape": (corrupt(lambda d: d["layers"][0].update(weights=[[0, 0, 0]])), "expected weights"),
            "weights not numbers": (corrupt(lambda d: d["layers"][0].update(weights="abc")), "invalid"),
            "NaN weight": (corrupt(lambda d: d["layers"][0]["biases"].__setitem__(0, float("nan"))), "finite"),
            "wrong activation": (corrupt(lambda d: d["layers"][0].update(activation="relu")), "doesn't match"),
            "input_size not the encoding": ((self.folder / "other_input.nn").read_text(), "expected 773 inputs"),
        }
        for reason, (text, message) in cases.items():
            with self.subTest(reason):
                (self.folder / "corrupted.nn").write_text(text)
                code, out, err = self.predict(BOARDS, "corrupted.nn")
                self.assertEqual((code, out), (84, ""))
                self.assertIn(message, err)

    def test_invalid_fens(self):
        for reason, fen in INVALID_FENS.items():
            with self.subTest(reason):
                code, out, err = self.predict([BOARDS[0], fen])
                self.assertEqual((code, out), (84, ""))
                self.assertRegex(err, r"^my_torch_analyzer: (board |\S+:)2: ")

    def test_invalid_labels(self):
        for label in filter(None, INVALID_LABELS):
            with self.subTest(label):
                code, out, err = self.predict([f"{BOARDS[0]} Check", f"{BOARDS[1]} {label}"])
                self.assertEqual((code, out), (84, ""))
                self.assertIn(f"board 2: unknown label '{label}'", err)

    def test_file_errors(self):
        (self.folder / "boards.txt").write_text(f"{BOARDS[0]}\n")
        (self.folder / "empty.txt").write_text("\n")
        (self.folder / "folder").mkdir()
        cases = {
            "missing CHESSFILE": ("basic.nn", "missing.txt", "missing.txt"),
            "empty CHESSFILE": ("basic.nn", "empty.txt", "no chessboard"),
            "CHESSFILE is a folder": ("basic.nn", "folder", "folder"),
            "LOADFILE is a folder": ("folder", "boards.txt", "folder"),
        }
        if os.geteuid() != 0:
            for name in ("basic.nn", "boards.txt"):
                locked = self.folder / f"locked_{name}"
                locked.write_bytes((self.folder / name).read_bytes())
                locked.chmod(0)
            cases["unreadable LOADFILE"] = ("locked_basic.nn", "boards.txt", "Permission denied")
            cases["unreadable CHESSFILE"] = ("basic.nn", "locked_boards.txt", "Permission denied")
        for reason, (loadfile, chessfile, message) in cases.items():
            with self.subTest(reason):
                code, out, err = run(["--predict", str(self.folder / loadfile), str(self.folder / chessfile)])
                self.assertEqual((code, out), (84, ""))
                self.assertIn(message, err)

    def test_built_analyzer(self):
        root = Path(__file__).resolve().parent.parent
        subprocess.run(["make", "-s", "-C", str(root)], check=True, capture_output=True)
        (self.folder / "boards.txt").write_text(f"{BOARDS[0]}\n")
        for network, code, out in (("basic.nn", 0, "Check\n"), ("missing.nn", 84, "")):
            with self.subTest(network):
                result = subprocess.run([root / "my_torch_analyzer", "--predict", network, "boards.txt"],
                                        cwd=self.folder, capture_output=True, text=True)
                self.assertEqual((result.returncode, result.stdout), (code, out))
                self.assertEqual(result.stderr.startswith("my_torch_analyzer: "), code == 84)


class TestTrain(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.loadfile = self.network("net.nn")
        self.write("boards.txt", [f"{board} {label}" for board, label in zip(BOARDS, LABELS)] * 10)

    def network(self, name, **changes):
        config = {"input_size": INPUT_SIZE, "layers": [{"size": 3, "activation": "softmax"}],
                  "classes": ["Nothing", "Check", "Checkmate"], "optimizer": "adam", "learning_rate": 0.05,
                  "batch_size": 8, "epochs": 20, "seed": 0, "validation_split": 0.25}
        Network(config | changes).save(self.folder / name)
        return self.folder / name

    def write(self, name, lines):
        (self.folder / name).write_text("".join(f"{line}\n" for line in lines))
        return str(self.folder / name)

    def train(self, *args, loadfile=None, chessfile="boards.txt"):
        return run(["--train", *args, str(loadfile or self.loadfile), str(self.folder / chessfile)])

    def test_train_then_predict(self):
        code, out, err = self.train()
        self.assertEqual((code, out), (0, ""))
        self.assertIn("epoch 20/20", err)
        self.assertIn("val_accuracy", err)
        self.assertEqual(run(["--predict", str(self.loadfile), self.write("test.txt", BOARDS)]),
                         (0, "Checkmate\nNothing\nNothing\nCheck\n", ""))

    def test_no_validation(self):
        code, out, err = self.train(loadfile=self.network("all.nn", validation_split=0))
        self.assertEqual((code, out), (0, ""))
        self.assertNotIn("val_accuracy", err)

    def test_save_keeps_loadfile(self):
        before = self.loadfile.read_bytes()
        self.assertEqual(self.train("--save", str(self.folder / "new.nn"))[0], 0)
        self.assertEqual(self.loadfile.read_bytes(), before)
        Network.load(self.folder / "new.nn")

    def test_savefile_in_missing_folder(self):
        before = self.loadfile.read_bytes()
        code, out, err = self.train("--save", str(self.folder / "missing" / "new.nn"))
        self.assertEqual((code, out), (84, ""))
        self.assertRegex(err, r"^my_torch_analyzer: \S+/missing/new.nn: folder '\S+/missing' doesn't exist\n$")
        self.assertEqual(self.loadfile.read_bytes(), before)

    def test_loadfile_unchanged_on_error(self):
        cases = {
            "missing label": (self.loadfile, [f"{BOARDS[0]} Check", BOARDS[1]], "bad.txt:2: missing label"),
            "invalid FEN": (self.loadfile, ["8/8/8 w - - 0 1 Check"], "board 1: "),
            "unknown label": (self.loadfile, [f"{BOARDS[0]} Check", f"{BOARDS[1]} Mate"], "board 2: unknown label"),
            "too few boards": (self.loadfile, [f"{BOARDS[0]} Check"], "not enough boards"),
            "bad validation_split": (self.network("split.nn", validation_split=1), [f"{BOARDS[0]} Check"],
                                     "validation_split"),
        }
        for reason, (loadfile, lines, message) in cases.items():
            with self.subTest(reason):
                self.write("bad.txt", lines)
                before = loadfile.read_bytes()
                code, out, err = self.train(loadfile=loadfile, chessfile="bad.txt")
                self.assertEqual((code, out), (84, ""))
                self.assertIn(message, err)
                self.assertEqual(loadfile.read_bytes(), before)

    def test_failed_save_keeps_loadfile(self):
        def broken_save(net, path):
            Path(path).write_text('{"half": ')
            raise OSError("disk full")

        before = self.loadfile.read_bytes()
        with mock.patch.object(Network, "save", broken_save):
            code, out, err = self.train()
        self.assertEqual((code, out), (84, ""))
        self.assertIn("disk full", err)
        self.assertEqual(self.loadfile.read_bytes(), before)
        self.assertEqual(sorted(path.name for path in self.folder.iterdir()), ["boards.txt", "net.nn"])


if __name__ == "__main__":
    unittest.main()
