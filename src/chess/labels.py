import numpy as np

CLASSES = ("Nothing", "Check", "Checkmate")
COLORS = ("White", "Black")
UNCOLORED = ("Nothing",)
ATTACKER = {"w": "Black", "b": "White"}


class LabelError(ValueError):
    pass


def encode_labels(labels, classes=CLASSES):
    indices = np.empty(len(labels), dtype=np.int64)
    for number, label in enumerate(labels, 1):
        indices[number - 1] = class_index(label, classes, number)
    return indices


def class_index(label, classes, number):
    words = label.split()
    if words and words[0] in classes:
        if len(words) == 1 or (len(words) == 2 and words[0] not in UNCOLORED and words[1] in COLORS):
            return list(classes).index(words[0])
    raise LabelError(f"board {number}: unknown label '{label}'")


def decode_labels(indices, sides, classes=CLASSES, full=False):
    names = [classes[index] for index in indices]
    if not full:
        return names
    return [name if name in UNCOLORED else f"{name} {ATTACKER[side]}"
            for name, side in zip(names, sides, strict=True)]
