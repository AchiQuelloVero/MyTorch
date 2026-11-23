#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.learning_curves import AXIS, COLORS, GRID, INK, SOFT_INK, SURFACE


def main():
    parser = argparse.ArgumentParser(description="Plot the results of the hyperparameter search.")
    parser.add_argument("--search", type=Path, default=Path("docs/search.md"), help="round 1 table")
    parser.add_argument("--retrain", type=Path, default=Path("docs/search_retrain.md"), help="round 2 table")
    parser.add_argument("--out", type=Path, default=Path("docs/search.png"))
    args = parser.parse_args()
    try:
        plot(read(args.search), read(args.retrain), args.out)
    except (OSError, ValueError, KeyError) as e:
        print(f"plot_search: {e}", file=sys.stderr)
        return 84
    return 0


def read(path):
    with open(path) as f:
        lines = f.read().splitlines()
    rows = [[cell.strip() for cell in line.strip("| ").split("|")] for line in lines if line.startswith("| ")]
    if len(rows) < 2:
        raise ValueError(f"{path}: no results table")
    return lines[0].split(": ", 1)[-1], [dict(zip(rows[0], row)) for row in rows[1:]]


def accuracy(row):
    return float(row["val_accuracy"].rstrip("%")) / 100


def style(axis, title, xlabel):
    axis.set_title(title, loc="left", color=INK, fontsize=11)
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.tick_params(colors=SOFT_INK, length=0, labelsize=9)
    axis.set_xlabel(xlabel, color=SOFT_INK, fontsize=9)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def plot(search, retrain, path):
    (search_title, trials), (retrain_title, retrained) = search, retrain
    by_number = {row["trial"]: row for row in trials}
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True, facecolor=SURFACE)
    for axis, key in zip(axes, ("dropout", "l2")):
        style(axis, f"Validation accuracy by {key}", key)
        for row in retrained:
            if row["trial"] in by_number:
                axis.plot([float(row[key])] * 2, [accuracy(by_number[row["trial"]]), accuracy(row)],
                          color=AXIS, linewidth=1, zorder=1)
        for rows, color, label in ((trials, COLORS[0], f"round 1: {search_title}"),
                                   (retrained, COLORS[1], f"round 2: {retrain_title}")):
            axis.plot([float(row[key]) for row in rows], [accuracy(row) for row in rows], "o", color=color,
                      markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        for row in retrained:
            if row["trial"] == "0":
                name, offset, align = "trial 0: basic.conf", (8, 0), ("left", "center")
            else:
                name, offset, align = f"trial {row['trial']}", (0, 8), ("center", "bottom")
            axis.annotate(name, (float(row[key]), accuracy(row)), xytext=offset, textcoords="offset points",
                          ha=align[0], va=align[1], color=SOFT_INK, fontsize=9, zorder=4)
    axes[1].set_xscale("log")
    axes[0].set_ylim(top=max(accuracy(row) for row in trials + retrained) + 0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Hyperparameter search (grey line: the same trial in both rounds)", x=0.01, y=0.985,
                    ha="left", color=INK, fontsize=13)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=2, frameon=False,
                  fontsize=9, labelcolor=SOFT_INK)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    sys.exit(main())
