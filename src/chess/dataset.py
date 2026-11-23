import numpy as np

from src.chess.encoding import canonical_key


class ChessFileError(ValueError):
    pass


def read_chessfile(path):
    with open(path) as f:
        lines = f.read().rstrip().splitlines()
    if not lines:
        raise ChessFileError(f"{path}: no chessboard")
    fens, labels = [], []
    for number, line in enumerate(lines, 1):
        words = line.split()
        if len(words) < 6:
            raise ChessFileError(f"{path}:{number}: expected a FEN with 6 fields, got '{line}'")
        fens.append(" ".join(words[:6]))
        labels.append(" ".join(words[6:]))
    return fens, labels


def deduplicate(fens, labels):
    seen = set()
    kept_fens, kept_labels = [], []
    for fen, label in zip(fens, labels, strict=True):
        key = canonical_key(fen)
        if key not in seen:
            seen.add(key)
            kept_fens.append(fen)
            kept_labels.append(label)
    return kept_fens, kept_labels


def split_indices(labels, fractions, seed):
    if any(fraction <= 0 for fraction in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError(f"split fractions must be positive and add up to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    parts = [[] for _ in fractions]
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        bounds = np.round(np.cumsum(fractions[:-1]) * len(indices)).astype(int)
        for part, chunk in zip(parts, np.split(indices, bounds)):
            part.append(chunk)
    return [rng.permutation(np.concatenate(part)) for part in parts]


def class_weights(y, nb_classes):
    counts = np.bincount(y, minlength=nb_classes)
    if len(counts) != nb_classes or not counts.all():
        raise ValueError(f"every class must appear in the labels, got counts {counts.tolist()}")
    return len(y) / (nb_classes * counts)
