#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess.dataset import read_chessfile
from src.chess.encoding import encode_fens
from src.chess.labels import encode_labels
from src.nn.network import Network

RUNS = (
    ("no regularization", 0.0, 0.0),
    ("l2 0.001", 0.001, 0.0),
    ("dropout 0.5", 0.0, 0.5),
    ("l2 0.001 + dropout 0.5", 0.001, 0.5),
)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SOFT_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def main():
    parser = argparse.ArgumentParser(description="Train several networks and plot their learning curves.")
    parser.add_argument("--config", type=Path, default=Path("config/basic.conf"))
    parser.add_argument("--splits", type=Path, default=Path("dataset/splits"), help="folder of train.txt and val.txt")
    parser.add_argument("--boards", type=int, help="only train on the first boards of train.txt")
    parser.add_argument("--epochs", type=int, help="overrides the epochs of the config")
    parser.add_argument("--out", type=Path, default=Path("docs"), help="folder of the PNG files")
    args = parser.parse_args()
    try:
        curves(args.config, args.splits, args.boards, args.epochs, args.out)
    except (OSError, ValueError) as e:
        print(f"learning_curves: {e}", file=sys.stderr)
        return 84
    return 0


def load(path, boards=None):
    fens, labels = read_chessfile(path)
    return encode_fens(fens[:boards]), encode_labels(labels[:boards])


def curves(config_path, splits, boards, epochs, out):
    with open(config_path) as f:
        config = json.load(f)
    if epochs is not None:
        config["epochs"] = epochs
    x, y = load(splits / "train.txt", boards)
    x_val, y_val = load(splits / "val.txt")
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, l2, dropout in RUNS:
        net = Network({**config, "l2": l2, "dropout": dropout})
        runs.append((name, net.train(x, y, x_val, y_val, verbose=True)))
    plot_train_val(runs, len(x), out / "curves_train_val.png")
    plot_networks(runs, len(x), out / "curves_networks.png")
    return runs


def best_record(history):
    return min(history, key=lambda record: record["val_loss"])


def values(history, key):
    return [record["epoch"] for record in history], [record[key] for record in history]


def style(axis, title, percent):
    axis.set_title(title, loc="left", color=INK, fontsize=11)
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.tick_params(colors=SOFT_INK, length=0, labelsize=9)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("epoch", color=SOFT_INK, fontsize=9)
    if percent:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def line(axis, history, key, color, label):
    axis.plot(*values(history, key), color=color, linewidth=2, label=label)
    last = history[-1]
    axis.annotate(label, (last["epoch"], last[key]), xytext=(6, 0), textcoords="offset points", va="center",
                  color=SOFT_INK, fontsize=9, annotation_clip=False, zorder=6,
                  bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})


def mark_best(axis, history, key, color):
    best = best_record(history)
    axis.plot(best["epoch"], best[key], "o", color=color, markersize=7, markeredgecolor=SURFACE,
              markeredgewidth=1.5, zorder=5)


def finish(figure, title, handles, labels, path):
    for axis in figure.axes:
        axis.set_xlim(left=0.5)
        axis.label_outer()
    figure.suptitle(title, x=0.01, y=0.985, ha="left", color=INK, fontsize=13)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=len(labels),
                  frameon=False, fontsize=9, labelcolor=SOFT_INK)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_train_val(runs, boards, path):
    shown = (runs[0], runs[-1])
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row", facecolor=SURFACE)
    for column, (name, history) in enumerate(shown):
        for row, (metric, percent) in enumerate((("loss", False), ("accuracy", True))):
            axis = axes[row][column]
            style(axis, f"{name}: {metric}", percent)
            line(axis, history, metric, COLORS[0], "train")
            line(axis, history, f"val_{metric}", COLORS[1], "validation")
            mark_best(axis, history, f"val_{metric}", COLORS[1])
    handles, labels = axes[0][0].get_legend_handles_labels()
    finish(figure, f"Training and validation, {boards:,} training boards (dot: epoch that is kept)",
           handles, labels, path)


def plot_networks(runs, boards, path):
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor=SURFACE)
    for axis, (key, title, percent) in zip(axes, (("val_loss", "Validation loss", False),
                                                  ("val_accuracy", "Validation accuracy", True))):
        style(axis, title, percent)
        for (name, history), color in zip(runs, COLORS):
            line(axis, history, key, color, name)
            mark_best(axis, history, key, color)
    handles, labels = axes[0].get_legend_handles_labels()
    finish(figure, f"Networks compared, {boards:,} training boards (dot: epoch that is kept)", handles, labels, path)


if __name__ == "__main__":
    sys.exit(main())
