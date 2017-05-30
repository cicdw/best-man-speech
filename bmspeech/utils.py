import numpy as np


def create_local_vec(keys, mapper):
    keys = [keys] if isinstance(keys, str) else keys
    lv = np.zeros(len(mapper))
    for key in keys:
        i = mapper[key]
        if i is not None:
            lv[mapper[key]] = 1
    return lv / lv.sum()
