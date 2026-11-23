#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess.dataset import ChessFileError, deduplicate, read_chessfile, split_indices  # noqa: E402

SPLITS = ("train", "val", "test")
FRACTIONS = (0.8, 0.1, 0.1)


def main():
    parser = argparse.ArgumentParser(description="Build the train, val and test files from the dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="folder of */*.txt files")
    parser.add_argument("--out", type=Path, default=Path("dataset/splits"), help="where to write the splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        build(args.dataset, args.out, args.seed)
    except (OSError, ValueError) as e:
        print(f"build_dataset: {e}", file=sys.stderr)
        return 84
    return 0


def build(dataset, out, seed):
    paths = sorted(path for path in dataset.glob("*/*.txt") if path.parent.resolve() != out.resolve())
    if not paths:
        raise ChessFileError(f"{dataset}: no */*.txt file")
    fens, labels = [], []
    for path in paths:
        path_fens, path_labels = read_chessfile(path)
        if "" in path_labels:
            raise ChessFileError(f"{path}:{path_labels.index('') + 1}: missing label")
        fens += path_fens
        labels += path_labels
    fens, unique_labels = deduplicate(fens, labels)
    print(f"{len(labels)} boards in {len(paths)} files, {len(labels) - len(fens)} duplicates removed")

    out.mkdir(parents=True, exist_ok=True)
    names = sorted(set(unique_labels))
    print(f"{'split':6}{'boards':>8}" + "".join(f"{name:>17}" for name in names))
    for split, indices in zip(SPLITS, split_indices(unique_labels, FRACTIONS, seed)):
        with open(out / f"{split}.txt", "w") as f:
            f.writelines(f"{fens[i]} {unique_labels[i]}\n" for i in indices)
        counts = Counter(unique_labels[i] for i in indices)
        print(f"{split:6}{len(indices):8}" + "".join(f"{counts[name]:17}" for name in names))


if __name__ == "__main__":
    sys.exit(main())
