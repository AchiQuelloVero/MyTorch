#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess.dataset import read_chessfile
from src.chess.encoding import encode_fens
from src.chess.labels import encode_labels
from src.nn.network import Network

HIDDEN = ((128,), (256,), (128, 32), (256, 64), (512, 128), (512, 256, 64))
BATCH_SIZES = (32, 64, 128, 256)
COLUMNS = ("trial", "hidden", "learning_rate", "batch_size", "dropout", "l2", "best epoch", "val_accuracy",
           "val_loss", "seconds")


def main():
    parser = argparse.ArgumentParser(description="Random search of the hyperparameters, scored on val.txt.")
    parser.add_argument("--config", type=Path, default=Path("config/basic.conf"), help="settings that don't change")
    parser.add_argument("--splits", type=Path, default=Path("dataset/splits"), help="folder of train.txt and val.txt")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--boards", type=int, default=100000, help="training boards of each trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", type=int, nargs="+", metavar="TRIAL",
                        help="retrain these trials on the whole train.txt instead of searching (0: --config as is)")
    parser.add_argument("--final-epochs", type=int, default=60, help="maximum epochs of the retrained trials")
    parser.add_argument("--out", type=Path, help="results table (default docs/search.md, docs/search_retrain.md)")
    parser.add_argument("--best", type=Path, default=Path("config/best.conf"), help="best retrained configuration")
    args = parser.parse_args()
    try:
        search(args)
    except (OSError, ValueError) as e:
        print(f"search: {e}", file=sys.stderr)
        return 84
    return 0


def load(path, boards=None):
    fens, labels = read_chessfile(path)
    return encode_fens(fens[:boards]), encode_labels(labels[:boards])


def significant(value):
    return float(f"{value:.2g}")


def sample(rng, base):
    hidden = HIDDEN[rng.integers(len(HIDDEN))]
    return {
        **base,
        "layers": [{"size": size, "activation": "relu"} for size in hidden] + [base["layers"][-1]],
        "learning_rate": significant(10 ** rng.uniform(-4, -2.5)),
        "batch_size": int(rng.choice(BATCH_SIZES)),
        "dropout": round(float(rng.uniform(0, 0.6)), 2),
        "l2": significant(10 ** rng.uniform(-5, -2)),
    }


def trial(number, config, x, y, x_val, y_val):
    start = time.perf_counter()
    net = Network(config)
    history = net.train(x, y, x_val, y_val)
    val_loss, val_accuracy = net.evaluate(x_val, y_val)
    best = min(history, key=lambda record: record["val_loss"])["epoch"]
    hidden = "-".join(str(layer["size"]) for layer in config["layers"][:-1])
    return (number, hidden, config["learning_rate"], config["batch_size"], config["dropout"], config["l2"],
            f"{best} of {len(history)}", f"{val_accuracy:.2%}", f"{val_loss:.4f}", round(time.perf_counter() - start))


def accuracy(row):
    return float(row[7].rstrip("%"))


def write(path, title, command, rows):
    with open(path, "w") as f:
        f.write(f"# {title}\n\n`{command}`: best validation accuracy first.\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n")
        f.writelines("| " + " | ".join(str(value) for value in row) + " |\n"
                     for row in sorted(rows, key=accuracy, reverse=True))


def draw(args, base, count):
    rng = np.random.default_rng(args.seed)
    return {0: dict(base)} | {number: {**sample(rng, base), "seed": args.seed + number} for number in range(1, count + 1)}


def run(configs, numbers, x, y, x_val, y_val, title, command, out):
    rows = []
    for number in numbers:
        rows.append(trial(number, configs[number], x, y, x_val, y_val))
        print(" ".join(f"{name} {value}" for name, value in zip(COLUMNS, rows[-1])), file=sys.stderr, flush=True)
        write(out, title, command, rows)
    return rows


def search(args):
    with open(args.config) as f:
        base = json.load(f)
    x_val, y_val = load(args.splits / "val.txt")
    if not args.retrain:
        x, y = load(args.splits / "train.txt", args.boards)
        command = f"scripts/search.py --trials {args.trials} --boards {args.boards} --seed {args.seed}"
        run(draw(args, base, args.trials), range(1, args.trials + 1), x, y, x_val, y_val,
            f"Hyperparameter search: {args.trials} trials on {args.boards} boards", command,
            args.out or Path("docs/search.md"))
        return
    if min(args.retrain) < 0:
        raise ValueError("trial numbers can't be negative")
    configs = draw(args, base, max(args.retrain))
    for number in args.retrain:
        configs[number]["epochs"] = args.final_epochs
    x, y = load(args.splits / "train.txt")
    command = (f"scripts/search.py --seed {args.seed} --retrain {' '.join(map(str, args.retrain))} "
               f"--final-epochs {args.final_epochs}")
    rows = run(configs, args.retrain, x, y, x_val, y_val,
               "Hyperparameter search: best trials retrained on the whole train.txt", command,
               args.out or Path("docs/search_retrain.md"))
    with open(args.best, "w") as f:
        json.dump(configs[max(rows, key=accuracy)[0]], f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
