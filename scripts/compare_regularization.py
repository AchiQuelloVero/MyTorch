#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess.dataset import read_chessfile
from src.chess.encoding import encode_fens
from src.chess.labels import encode_labels
from src.nn.network import Network

RUNS = (
    ("nothing", 0.0, 0.0),
    ("l2 0.0001", 0.0001, 0.0),
    ("l2 0.001", 0.001, 0.0),
    ("dropout 0.2", 0.0, 0.2),
    ("dropout 0.5", 0.0, 0.5),
    ("l2 0.0001 + dropout 0.2", 0.0001, 0.2),
    ("l2 0.001 + dropout 0.5", 0.001, 0.5),
)


def main():
    parser = argparse.ArgumentParser(description="Train with and without L2 and dropout, compare the results.")
    parser.add_argument("--config", type=Path, default=Path("config/basic.conf"))
    parser.add_argument("--splits", type=Path, default=Path("dataset/splits"), help="folder of train.txt and val.txt")
    parser.add_argument("--boards", type=int, help="only train on the first boards of train.txt")
    parser.add_argument("--epochs", type=int, help="overrides the epochs of the config")
    args = parser.parse_args()
    try:
        compare(args.config, args.splits, args.boards, args.epochs)
    except (OSError, ValueError) as e:
        print(f"compare_regularization: {e}", file=sys.stderr)
        return 84
    return 0


def load(path, boards=None):
    fens, labels = read_chessfile(path)
    return encode_fens(fens[:boards]), encode_labels(labels[:boards])


def compare(config_path, splits, boards, epochs):
    with open(config_path) as f:
        config = json.load(f)
    if epochs is not None:
        config["epochs"] = epochs
    x, y = load(splits / "train.txt", boards)
    x_val, y_val = load(splits / "val.txt")

    print(f"{'':26}{'best epoch':>12}{'train':>9}{'val':>9}{'gap':>9}{'val_loss':>10}")
    for name, l2, dropout in RUNS:
        net = Network({**config, "l2": l2, "dropout": dropout})
        history = net.train(x, y, x_val, y_val, verbose=True)
        best = min(history, key=lambda record: record["val_loss"])["epoch"]
        accuracy = net.evaluate(x, y)[1]
        val_loss, val_accuracy = net.evaluate(x_val, y_val)
        print(f"{name:26}{f'{best} of {len(history)}':>12}{accuracy:9.2%}{val_accuracy:9.2%}"
              f"{accuracy - val_accuracy:9.2%}{val_loss:10.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
