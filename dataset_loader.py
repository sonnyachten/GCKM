from typing import List, Union

import torch_geometric.transforms as T
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import BaseTransform

from definitions import *
from utils import index_to_mask


def DataLoader(name, preprocess=False):
    name = name.lower()
    if isinstance(preprocess, bool):
        if preprocess:
            transform = T.NormalizeFeatures()
        else:
            transform = None
    elif preprocess is None:
        transform = None
    elif isinstance(preprocess, str):
        if preprocess in ['standardize']:
            transform = StandardizeFeatures()
        elif preprocess in ['no', 'none']:
            transform = None
        else:
            ValueError("preprocess should be 'no', 'normalize', or 'standardize'")
    else:
        raise AssertionError

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(DATA_DIR, name, transform=transform)
        data = dataset[0]
    elif name in ['chameleon', 'squirrel']:
        preProcDs = WikipediaNetwork(
            root=DATA_DIR, name=name, geom_gcn_preprocess=False, transform=transform)
        dataset = WikipediaNetwork(
            root=DATA_DIR, name=name, geom_gcn_preprocess=True, transform=transform)
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=DATA_DIR, transform=transform)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data = dataset[0]
        data.train_mask = index_to_mask(train_idx, data.num_nodes)
        data.val_mask = index_to_mask(valid_idx, data.num_nodes)
        data.test_mask = index_to_mask(test_idx, data.num_nodes)
        # data = dataset[0]
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return data, dataset.num_classes, dataset.num_features


class StandardizeFeatures(BaseTransform):
    r"""Standardizes the attributes given in :obj:`attrs` column-wise

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """

    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value - value.mean(dim=0)
                value.div_(value.std(dim=0, keepdim=True).clamp_(min=1e-12))
                store[key] = value
        return data
