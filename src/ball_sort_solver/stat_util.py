import numpy as np


def moving_max(values):
    max_val = None
    monotone_values = []
    for val in values:
        if max_val is None or val > max_val:
            max_val = val
        monotone_values.append(max_val)
    return np.array(monotone_values)


def moving_mean(values, n):
    return np.convolve(values, np.ones(n), mode='valid') / n


def safe_mean(values):
    if len(values) == 0:
        return 0
    return np.mean(values)
