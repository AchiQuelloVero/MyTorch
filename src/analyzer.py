#!/usr/bin/env python3
import os
import sys
from dataclasses import dataclass

import numpy as np

from src.chess.dataset import ChessFileError, read_chessfile, split_indices
from src.chess.encoding import encode_fens
from src.chess.labels import decode_labels, encode_labels
from src.nn.network import Network

USAGE = """USAGE
        ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
        --train     Launch the neural network in training mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.

        --predict   Launch the neural network in prediction mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output.

        --save      Save neural network into SAVEFILE. Only works in train mode.

        LOADFILE    File containing an artificial neural network
        CHESSFILE   File containing chessboards"""


class UsageError(Exception):
    pass


@dataclass(frozen=True)
class Arguments:
    mode: str
    loadfile: str
    chessfile: str
    savefile: str


def parse_args(args):
    mode, savefile, files = None, None, []
    args = iter(args)
    for arg in args:
        if arg in ("--predict", "--train"):
            if mode is not None:
                raise UsageError("give only one of --predict and --train")
            mode = arg[2:]
        elif arg == "--save":
            if savefile is not None:
                raise UsageError("--save given twice")
            savefile = next(args, None)
            if savefile is None:
                raise UsageError("--save needs a SAVEFILE")
        elif arg.startswith("-"):
            raise UsageError(f"unknown option '{arg}'")
        else:
            files.append(arg)
    if mode is None:
        raise UsageError("give one of --predict and --train")
    if savefile is not None and mode != "train":
        raise UsageError("--save only works with --train")
    if len(files) != 2:
        raise UsageError(f"expected LOADFILE and CHESSFILE, got {len(files)} file(s)")
    return Arguments(mode, files[0], files[1], savefile or files[0])


def predict(arguments):
    net = Network.load(arguments.loadfile)
    fens, labels = read_chessfile(arguments.chessfile)
    predicted = net.predict(encode_fens(fens)).argmax(axis=1)
    expected = encode_labels(labels, net.classes) if all(labels) else None
    sides = [fen.split()[1] for fen in fens]
    print("\n".join(decode_labels(predicted, sides, net.classes, net.config.get("output") == "full")))
    if expected is not None:
        print(f"accuracy {np.mean(predicted == expected):.2%} on {len(fens)} boards", file=sys.stderr)


def train(arguments):
    folder = os.path.dirname(arguments.savefile) or "."
    if not os.path.isdir(folder):
        raise ValueError(f"{arguments.savefile}: folder '{folder}' doesn't exist")
    net = Network.load(arguments.loadfile)
    fens, labels = read_chessfile(arguments.chessfile)
    if "" in labels:
        raise ChessFileError(f"{arguments.chessfile}:{labels.index('') + 1}: missing label")
    x, y = encode_fens(fens), encode_labels(labels, net.classes)
    validation = float(net.config.get("validation_split", 0.1))
    if not 0 <= validation < 1:
        raise ValueError(f"validation_split must be in [0, 1), got {validation}")
    if validation == 0:
        net.train(x, y, verbose=True)
    else:
        kept, held = split_indices(y, (1 - validation, validation), net.config.get("seed"))
        if len(kept) == 0 or len(held) == 0:
            raise ChessFileError(f"{arguments.chessfile}: not enough boards to hold {validation:.0%} for validation")
        net.train(x[kept], y[kept], x[held], y[held], verbose=True)
    save(net, arguments.savefile)


def save(net, path):
    temporary = f"{path}.tmp"
    try:
        net.save(temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def main(args):
    if args in (["--help"], ["-h"]):
        print(USAGE)
        return 0
    try:
        arguments = parse_args(args)
    except UsageError as e:
        print(f"my_torch_analyzer: {e}\nTry './my_torch_analyzer --help'.", file=sys.stderr)
        return 84
    try:
        (train if arguments.mode == "train" else predict)(arguments)
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except (OSError, ValueError) as e:
        print(f"my_torch_analyzer: {e}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
