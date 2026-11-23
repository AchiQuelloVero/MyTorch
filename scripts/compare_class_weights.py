#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess.dataset import class_weights, read_chessfile
from src.chess.encoding import encode_fens
from src.chess.labels import CLASSES, encode_labels
from src.nn.network import Network


def main():
    parser = argparse.ArgumentParser(description="Train with and without class weights, compare per-class accuracy.")
    parser.add_argument("--config", type=Path, default=Path("config/basic.conf"))
    parser.add_argument("--splits", type=Path, default=Path("dataset/splits"), help="folder of train.txt and val.txt")
    parser.add_argument("--epochs", type=int, help="overrides the epochs of the config")
    args = parser.parse_args()
    try:
        compare(args.config, args.splits, args.epochs)
    except (OSError, ValueError) as e:
        print(f"compare_class_weights: {e}", file=sys.stderr)
        return 84
    return 0


def load(path):
    fens, labels = read_chessfile(path)
    return encode_fens(fens), encode_labels(labels)


def compare(config_path, splits, epochs):
    with open(config_path) as f:
        config = json.load(f)
    if epochs is not None:
        config["epochs"] = epochs
    x, y = load(splits / "train.txt")
    x_val, y_val = load(splits / "val.txt")
    weights = class_weights(y, len(CLASSES))
    print("class weights: " + ", ".join(f"{name} {weight:.3f}" for name, weight in zip(CLASSES, weights)))

    print(f"{'':10}{'accuracy':>10}" + "".join(f"{name:>11}" for name in CLASSES))
    for name, run_weights in (("without", None), ("with", weights)):
        net = Network(config)
        net.train(x, y, x_val, y_val, class_weights=run_weights, verbose=True)
        predicted = net.predict(x_val).argmax(axis=1)
        per_class = [np.mean(predicted[y_val == index] == index) for index in range(len(CLASSES))]
        print(f"{name:10}{np.mean(predicted == y_val):10.2%}" + "".join(f"{value:11.2%}" for value in per_class))


if __name__ == "__main__":
    sys.exit(main())
