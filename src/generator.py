#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

from src.nn.network import Network

USAGE = """USAGE
        ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
        config_file_i   Configuration file containing description of a neural network we want to generate.
        nb_i            Number of neural networks to generate based on the configuration file."""


class UsageError(Exception):
    pass


def parse_args(args):
    if not args or len(args) % 2:
        raise UsageError("expected pairs of config_file and nb")
    pairs = []
    for config_file, nb in zip(args[::2], args[1::2]):
        if not nb.isdigit() or int(nb) < 1:
            raise UsageError(f"nb must be a positive integer, got '{nb}'")
        pairs.append((config_file, int(nb)))
    return pairs


def load_config(path):
    with open(path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not a JSON file ({e})")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: the configuration must be a JSON object")
    seed = config.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError(f"{path}: seed must be a positive integer")
    try:
        Network(config)
    except ValueError as e:
        raise ValueError(f"{path}: {e}")
    return config


def seeds(config, nb, rng):
    if config.get("seed") is None:
        return [int(rng.integers(2 ** 31)) for _ in range(nb)]
    return [config["seed"] + i for i in range(nb)]


def generate(config_file, config, nb, rng):
    name = Path(config_file).stem
    paths = []
    for number, seed in enumerate(seeds(config, nb, rng), 1):
        path = f"{name}_{number}.nn"
        Network({**config, "seed": seed}).save(path)
        paths.append(path)
    return paths


def main(args):
    if args in (["--help"], ["-h"]):
        print(USAGE)
        return 0
    try:
        pairs = parse_args(args)
        configs = [(path, load_config(path), nb) for path, nb in pairs]
        rng = np.random.default_rng()
        for path, config, nb in configs:
            generate(path, config, nb, rng)
    except UsageError as e:
        print(f"my_torch_generator: {e}\nTry './my_torch_generator --help'.", file=sys.stderr)
        return 84
    except (OSError, ValueError) as e:
        print(f"my_torch_generator: {e}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
