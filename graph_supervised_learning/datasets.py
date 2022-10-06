import os
import argparse
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit, Reddit2, PPI
from torch_geometric.data.data import Data
import numpy as np
from typing import List

def load_data(name:str, path:str) -> Data:
    '''
    :param name: {Reddit, Reddit2, ogbn-arxiv, PPI, ogbn-products}
    :return: Data
    '''
    if name == 'Reddit':
        dataset = Reddit(root=os.path.join(path, 'Reddit'))
        data = dataset[0]
        data.x = data.x[:, 1:] # remove first feature, it has different scale
        return data
    elif name == 'Reddit2':
        dataset = Reddit2(root=os.path.join(path, 'Reddit2'))
        data = dataset[0]
        data.x = data.x[:, 2:] # remove first two features, they have different scale
        return data
    elif name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=name, root=os.path.join(path, name))
        data = dataset[0]
        idx_split = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[idx_split['train']] = 1
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[idx_split['valid']] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[idx_split['test']] = 1
        data.y = data.y.flatten()
        # Make ogbn-arxiv undirected.
        edge_index = data.edge_index
        tail_head = edge_index[:, edge_index[0] < edge_index[1]]
        head_tail = edge_index[:, edge_index[1] < edge_index[0]][[1, 0]]
        edges = torch.cat((tail_head, head_tail), dim=1)
        _, unique_indices = np.unique((edges[1] * (edges.max() + 1) + edges[0]).numpy(), return_index=True)
        unique_indices = torch.tensor(unique_indices)
        new_edges = edges[:, unique_indices]
        new_edge_index = torch.cat((new_edges, new_edges[[1, 0]]), dim=1)
        data.edge_index = new_edge_index
        return data
    elif name == 'ogbn-products':
        dataset = PygNodePropPredDataset(name=name, root=os.path.join(path, name))
        data = dataset[0]
        idx_split = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[idx_split['train']] = 1
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[idx_split['valid']] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[idx_split['test']] = 1
        data.y = data.y.flatten()
        return data
    elif name == 'PPI':
        root = os.path.join(path, 'PPI')
        dataset = [*PPI(root=root, split='train'), *PPI(root=root, split='val'), *PPI(root=root, split='test')]
        offsets = [0] + list(np.array([data.x.shape[0] for data in dataset]).cumsum())[:-1]
        num_train_nodes = np.array([data.x.shape[0] for data in dataset])[0:20].sum()
        num_val_nodes = np.array([data.x.shape[0] for data in dataset])[20:22].sum()
        for i in range(len(offsets)): dataset[i].edge_index += offsets[i]
        x = [data.x for data in dataset]
        x = torch.cat(x, dim=0)
        y = [data.y for data in dataset]
        y = torch.cat(y, dim=0)
        edge_index = [data.edge_index for data in dataset]
        edge_index = torch.cat(edge_index, dim=1)
        train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        train_mask[:num_train_nodes] = 1
        val_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        val_mask[num_train_nodes:num_train_nodes + num_val_nodes] = 1
        test_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        test_mask[num_train_nodes + num_val_nodes:] = 1
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data
    else:
        raise ValueError('Incorrect dataset: ' + name)

class Dataset:
    def __init__(self, data_list:List[Data], num_features:int, num_classes:int, task_type:str):
        self.__list = data_list
        self.__num_features = num_features
        self.__num_classes = num_classes
        assert task_type in ['s', 'm'], "task_type should be one of {'s', 'm'}"
        self.__task_type = task_type # {s or m}

    def __len__(self) -> int:
        return len(self.__list)

    def __getitem__(self, idx:int) -> Data:
        return self.__list[idx]

    @property
    def num_features(self) -> int:
        return self.__num_features

    @property
    def num_node_features(self) -> int:
        return self.__num_features

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def task_type(self) -> str:
        return self.__task_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Datasets')
    parser.add_argument('--output', default=None, type=str, help='The path to directory where datasets will be sotred.')
    args = parser.parse_args()

    if args.output is not None:
        for name in ['Reddit', 'Reddit2', 'ogbn-arxiv', 'ogbn-products']:
            data = load_data(name, args.output)
            dataset = Dataset([data], num_features=data.x.shape[1], num_classes=data.y.max().item() + 1, task_type='s') # (s)
            torch.save(dataset, os.path.join(args.output, name, 'ref.pt'))

        for name in ['PPI']:
            data = load_data(name, args.output)
            dataset = Dataset([data], num_features=data.x.shape[1], num_classes=data.y.shape[1], task_type='m') # (m)
            torch.save(dataset, os.path.join(args.output, name, 'ref.pt'))











