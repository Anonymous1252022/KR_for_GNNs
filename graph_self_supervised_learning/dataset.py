import os

import torch_geometric.data
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig
from torch_geometric.datasets import Planetoid, Reddit, Reddit2, Flickr, PPI
import torch


def load_cora(cfg:DictConfig) -> torch_geometric.data.Dataset:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    return Planetoid(root=os.path.join(data_dir, "cora"), name="Cora", **ds_params)


def load_citeseer(cfg:DictConfig) -> torch_geometric.data.Dataset:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    return Planetoid(root=os.path.join(data_dir, "citeseer"), name="CiteSeer", **ds_params)


def load_pubmed(cfg:DictConfig) -> torch_geometric.data.Dataset:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    return Planetoid(root=os.path.join(data_dir, "pubmed"), name="PubMed", **ds_params)


def load_reddit(cfg:DictConfig) -> torch_geometric.data.Dataset:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    return Reddit(root=os.path.join(data_dir, "reddit"), **ds_params)


def load_reddit2(cfg:DictConfig) -> torch_geometric.data.Dataset:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    return Reddit2(root=os.path.join(data_dir, "reddit2"), **ds_params)


def load_ppi(cfg:DictConfig) -> torch_geometric.data.Data:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    data_dir = os.path.join(data_dir, "ppi")
    train_ds = PPI(root=data_dir,split="train", **ds_params)
    val_ds = PPI(root=data_dir, split="val", **ds_params)
    test_ds = PPI(root=data_dir, split="test", **ds_params)

    # Build masks
    data_map = {"train_mask": [],
                "val_mask": [],
                "test_mask": []}
    for curr_ds, relevant_mask in ((train_ds, "train_mask"), (val_ds, "val_mask"), (test_ds, "test_mask")):
        for data in curr_ds:
            data.val_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
            data.train_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
            data.test_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
            setattr(data, relevant_mask, torch.ones((data.x.size(0),), dtype=torch.bool))
            data_map[relevant_mask].append(data)

    # Merge graphs
    data = torch_geometric.data.Batch.from_data_list(data_map["train_mask"] + data_map["val_mask"] + data_map["test_mask"])
    return data

def load_ogb_dataset(name, ds_params, data_dir):
    dataset = PygNodePropPredDataset(name=name,
                                     root=data_dir,
                                     **ds_params)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
    data.train_mask[split_idx["train"]] = True
    data.val_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
    data.val_mask[split_idx["valid"]] = True
    data.test_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
    data.test_mask[split_idx["test"]] = True
    data.y = data.y.flatten()
    return data


def load_ogbn_arxiv(cfg:DictConfig) -> torch_geometric.data.Data:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    data = load_ogb_dataset("ogbn-arxiv", ds_params, data_dir)
    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, None, num_nodes=data.x.size(0))
    return data


def load_ogbn_products(cfg:DictConfig) -> torch_geometric.data.Data:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    data = load_ogb_dataset("ogbn-products", ds_params, data_dir)
    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, None, num_nodes=data.x.size(0))
    return data


def load_flickr(cfg:DictConfig) -> torch_geometric.data.Dataset:
    ds_params = cfg.dataset.params
    data_dir = cfg.enviroment.data_dir
    dataset = Flickr(root=os.path.join(data_dir, "flickr"), **ds_params)
    return dataset

def get_dataset(config: DictConfig) -> torch_geometric.data.Data:
    dataset_map = {"cora": load_cora,
                   "citeseer": load_citeseer,
                   "pubmed": load_pubmed,
                   "reddit": load_reddit,
                   "reddit2": load_reddit2,
                   "ogbn-arxiv": load_ogbn_arxiv,
                   "ogbn-products": load_ogbn_products,
                   "flickr": load_flickr,
                   "ppi": load_ppi}

    try:
        ds_name = config.dataset.name
        dataset = dataset_map[ds_name](config)
    except KeyError:
        raise RuntimeError(f"Invalid dataset received: {ds_name}, available: {list(dataset_map.keys())}")

    if isinstance(dataset, (torch_geometric.data.Data, tuple, list)):
        data = dataset
    else:
        data = dataset.data

    if config.dataset.normalize_features:
        x = data.x
        mean = torch.mean(x, dim=-1).unsqueeze(dim=1)
        std = torch.std(x, dim=-1).unsqueeze(dim=1)
        std[torch.abs(std) < 1e-4] = 1
        x_normed = (x-mean) / std
        data.x = x_normed

    if config.dataset.remove_self_loops:
        data.edge_index, _ = torch_geometric.utils.remove_self_loops(data.edge_index)

    return data