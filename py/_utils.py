import numpy as np


def load_data(fname, size, dtype="f4") -> np.ndarray:
    return np.fromfile(fname, dtype=dtype).reshape(size)


def extrema(d: np.ndarray):
    return np.min(d), np.max(d)


def entropy(d: np.ndarray):
    total = np.sum(d)
    res = 0
    for i in d:
        if i != 0:
            p = i / total
            res += -np.log2(p) * p
    return res
