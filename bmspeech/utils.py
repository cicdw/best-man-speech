import numpy as np


def print_top(vector, mapper=None, n=10, display_vals=False):
    ranks = list(reversed(np.argsort(vector)))
    for rank, idx in enumerate(ranks[:n]):
        values = ': {}'.format(vector[idx]) if display_vals else ''
        print('{0}.) {1}'.format(rank + 1, mapper[idx]) + values)


def create_local_vec(keys, mapper):
    keys = [keys] if isinstance(keys, str) else keys
    lv = np.zeros(len(mapper))
    for key in keys:
        i = mapper[key]
        if i is not None:
            lv[mapper[key]] = 1
    return lv / lv.sum()


def generate_counts_from_weights(eig, worker, stop_words={}, scale=1e3):
    fake_counts = {}
    fake_vals = (eig * scale).round().tolist()
    for idx, val in enumerate(fake_vals):
        word = worker[idx]
        if word in stop_words:
            pass
        else:
            fake_counts[word] = max(val, 1)
    return fake_counts
