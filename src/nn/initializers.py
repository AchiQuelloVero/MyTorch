import numpy as np


def xavier(rng, fan_in, fan_out):
    return rng.normal(0, np.sqrt(2 / (fan_in + fan_out)), (fan_in, fan_out))


def he(rng, fan_in, fan_out):
    return rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))


def lecun(rng, fan_in, fan_out):
    return rng.normal(0, np.sqrt(1 / fan_in), (fan_in, fan_out))


INITIALIZERS = {
    "xavier": xavier,
    "he": he,
    "lecun": lecun,
}
