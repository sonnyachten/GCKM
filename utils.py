import collections

import numpy as np

from definitions import *


def flatten_dict(d, parent_key='', sep='_', prefix='eval_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep, prefix=prefix).items())
        else:
            items.append((prefix + new_key, v))
    return dict(items)


def float_format(f: float) -> str:
    return "%+.4e" % f


def dot_mm(A, B):
    return torch.trace(torch.mm(A.t(), B))


def merge_two_dicts(x, y):
    return {**x, **y}


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def fixed_splits(data, num_classes, percls_trn, val_lb, name):
    seed = 42
    # seed = 1941488137
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data
