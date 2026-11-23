#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess.dataset import ChessFileError, read_chessfile
from src.chess.encoding import encode_fens
from src.chess.labels import decode_labels, encode_labels
from src.metrics import confusion_matrix, scores
from src.nn.network import Network

FULL = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
PIECES = (("2-9 pieces", 2, 9), ("10-19 pieces", 10, 19), ("20-32 pieces", 20, 32))
RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")


def main():
    parser = argparse.ArgumentParser(description="Accuracy, confusion matrices and per-class scores of a network.")
    parser.add_argument("network", type=Path, help="network file (.nn)")
    parser.add_argument("--chessfile", type=Path, default=Path("dataset/splits/test.txt"), help="labeled boards")
    parser.add_argument("--plot", type=Path, help="also draw the confusion matrix and the recall by pieces (PNG)")
    args = parser.parse_args()
    try:
        results = evaluate(args.network, args.chessfile)
        print(report(f"scripts/evaluate.py {args.network} --chessfile {args.chessfile}", *results))
        if args.plot:
            plot(*results, args.plot)
    except (OSError, ValueError) as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 84
    return 0


def evaluate(network, chessfile):
    net = Network.load(network)
    fens, labels = read_chessfile(chessfile)
    if "" in labels:
        raise ChessFileError(f"{chessfile}:{labels.index('') + 1}: missing label")
    expected = encode_labels(labels, net.classes)
    predicted = net.predict(encode_fens(fens)).argmax(axis=1)
    sides = [fen.split()[1] for fen in fens]
    full = [[FULL.index(name) for name in decode_labels(indices, sides, net.classes, full=True)]
            for indices in (expected, predicted)]
    pieces = np.array([sum(char.isalpha() for char in fen.split()[0]) for fen in fens])
    return net.classes, expected, predicted, full, pieces


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def confusion_table(names, matrix):
    return table(("expected / predicted", *names), [(name, *row) for name, row in zip(names, matrix.tolist())])


def by_pieces(classes, expected, predicted, pieces):
    for name, low, high in PIECES:
        kept = (pieces >= low) & (pieces <= high)
        if kept.any():
            yield name, int(kept.sum()), scores(confusion_matrix(expected[kept], predicted[kept], len(classes)))


def report(command, classes, expected, predicted, full, pieces):
    matrix = confusion_matrix(expected, predicted, len(classes))
    result = scores(matrix)
    per_class = [(name, matrix[index].sum(), f"{result['precision'][index]:.2%}", f"{result['recall'][index]:.2%}",
                  f"{result['f1'][index]:.4f}") for index, name in enumerate(classes)]
    pieces_rows = [(name, boards, f"{score['accuracy']:.2%}", f"{score['macro_f1']:.4f}",
                    *(f"{recall:.2%}" for recall in score["recall"]))
                   for name, boards, score in by_pieces(classes, expected, predicted, pieces)]
    return "\n\n".join((
        f"`{command}`: {len(expected)} boards, accuracy **{result['accuracy']:.2%}**, "
        f"macro-F1 **{result['macro_f1']:.4f}**",
        table(("class", "boards", "precision", "recall", "F1"), per_class),
        "Confusion matrix (rows: expected, columns: predicted):\n\n" + confusion_table(classes, matrix),
        "Confusion matrix of the full output:\n\n" + confusion_table(FULL, confusion_matrix(*full, len(FULL))),
        "By number of pieces:\n\n" + table(("pieces", "boards", "accuracy", "macro-F1",
                                            *(f"recall {name}" for name in classes)), pieces_rows),
    ))


def plot(classes, expected, predicted, full, pieces, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import PercentFormatter

    from scripts.learning_curves import AXIS, COLORS, GRID, INK, SOFT_INK, SURFACE

    matrix = confusion_matrix(expected, predicted, len(classes))
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": (1, 1.3)},
                                         facecolor=SURFACE)
    left.imshow(shares, cmap=LinearSegmentedColormap.from_list("blue", RAMP), vmin=0, vmax=1)
    for row in range(len(classes)):
        for column in range(len(classes)):
            left.text(column, row, f"{shares[row, column]:.1%}\n{matrix[row, column]:,}", ha="center", va="center",
                      fontsize=10, color=SURFACE if shares[row, column] > 0.5 else INK)
    left.set_xticks(range(len(classes)), classes)
    left.set_yticks(range(len(classes)), classes)
    left.set_xlabel("predicted", color=SOFT_INK, fontsize=9)
    left.set_ylabel("expected", color=SOFT_INK, fontsize=9)
    left.set_title("Confusion matrix (share of each expected class)", loc="left", color=INK, fontsize=11)
    left.tick_params(colors=SOFT_INK, length=0, labelsize=9)
    left.spines[:].set_visible(False)

    rows = list(by_pieces(classes, expected, predicted, pieces))
    width = 0.8 / len(classes)
    for index, (name, color) in enumerate(zip(classes, COLORS)):
        positions = [group + (index - (len(classes) - 1) / 2) * width for group in range(len(rows))]
        values = [score["recall"][index] for _, _, score in rows]
        bars = right.bar(positions, values, width * 0.92, color=color, label=name, zorder=3)
        right.bar_label(bars, [f"{value:.1%}" for value in values], padding=3, fontsize=8, color=SOFT_INK)
    right.set_xticks(range(len(rows)), [f"{name}\n{boards:,} boards" for name, boards, _ in rows])
    right.set_ylim(0, 1.08)
    right.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    right.set_title("Recall of each class by number of pieces", loc="left", color=INK, fontsize=11)
    right.set_facecolor(SURFACE)
    right.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    right.spines[["top", "right", "left"]].set_visible(False)
    right.spines["bottom"].set_color(AXIS)
    right.tick_params(colors=SOFT_INK, length=0, labelsize=9)
    figure.legend(*right.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.995, 0.93),
                  frameon=False, fontsize=9, labelcolor=SOFT_INK, ncol=len(classes))
    figure.suptitle(f"Evaluation on {len(expected):,} boards never used for training", x=0.01, y=0.98, ha="left",
                    color=INK, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    sys.exit(main())
