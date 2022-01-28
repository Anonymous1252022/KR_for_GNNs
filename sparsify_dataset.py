import os
import argparse
import torch
from torch_geometric.data.data import Data
from dataset import Dataset
from tqdm import tqdm
import numpy as np
from torch_scatter import scatter
from typing import Union, List
'''
    Sparsifiers: All graphs are assumed to be directed.
        - empty: Drops all edges from the graph.
        - rand: Randomly drops some portion of edges from the graph.
        - max_d: Computes maximal input degree according a given portion, and for each node drops edges if they exceeds maximal input degree computed above.
'''
def empty(path:str) -> type(None):
    print('Drop all edges.')
    dataset = torch.load(os.path.join(path, 'ref.pt'))
    dataset[0].edge_index = dataset[0].edge_index.T[[]].T
    torch.save(dataset, os.path.join(path, 'empty.pt'))

def rand(path:str, drop_edge_portions:List[float]) -> type(None):
    '''
        Input should be undirected graph.
    '''
    dataset = torch.load(os.path.join(path, 'ref.pt'))
    data = dataset[0]
    edge_index = data.edge_index

    for drop_edge_portion in drop_edge_portions:
        print('Random drop: ' + str(drop_edge_portion * 100) + '%')
        edge_selection = torch.tensor(np.random.binomial(1, 1.0 - drop_edge_portion, size=edge_index.shape[1])).type(torch.bool)
        new_edge_index = edge_index[:, edge_selection]
        new_data = Data(x=data.x, edge_index=new_edge_index, y=data.y)
        new_data.train_mask = data.train_mask
        new_data.val_mask = data.val_mask
        new_data.test_mask = data.test_mask
        new_dataset = Dataset([new_data], num_features=dataset.num_features, num_classes=dataset.num_classes, task_type=dataset.task_type)
        torch.save(new_dataset, os.path.join(path, 'rand_p' + str(drop_edge_portion).replace('.', '').replace('10', '1').replace('00', '0') + '.pt'))

def graph_forward(x:torch.Tensor, edge_index:torch.Tensor, edge_weights:Union[torch.Tensor, type(None)]=None, reduce:str='sum', keep_x_dim:bool=False) -> torch.Tensor:
    src = x[edge_index[0]] # Input nodes.

    if edge_weights is not None:
        src = src * edge_weights.reshape(-1, 1)

    index = edge_index[1] # Output node indices.
    if keep_x_dim:
        return scatter(src.T, index, reduce=reduce, dim_size=x.shape[0]).T
    else:
        return scatter(src.T, index, reduce=reduce).T

def get_t(d_in:np.ndarray, p:float) -> int:
    d_max = d_in.max()
    t_left = 0
    t_right = d_max
    for _ in range(100):
        t = t_left + (t_right - t_left) / 2
        p_tild = (d_in[d_in < t].sum() + (d_in >= t).sum() * t) / d_in.sum()
        if p_tild < p:
            t_left = t
        elif p_tild > p:
            t_right = t
        else:
            break

    return int(np.round(t))

def max_d(path:str, drop_edge_portions:List[float]) -> type(None):
    dataset = torch.load(os.path.join(path, 'ref.pt'))
    data = dataset[0]
    x = torch.ones(data.x.shape[0])
    d_in = graph_forward(x, data.edge_index).numpy().astype(np.int64)

    heads = data.edge_index[1]
    tmp = list(torch.stack((heads, torch.arange(heads.shape[0]))).T.numpy())
    tmp = np.array(sorted(tmp, key=lambda e: e[0]))
    indices = torch.tensor(tmp).T[1]


    for drop_edge_portion in drop_edge_portions:
        print('Maximal input degree drop: ' + str(drop_edge_portion * 100) + '%')
        t = get_t(d_in, p=1.0 - drop_edge_portion)
        selector = []
        for node_id in tqdm(range(d_in.shape[0])):
            if d_in[node_id] > t:
                selector.append(np.random.choice(a=t * [True] + (d_in[node_id] - t) * [False], size=d_in[node_id], replace=False))  # select only t
            else:
                selector.append(np.array(d_in[node_id] * [True]))  # select all
        selector = np.concatenate(selector).astype(np.bool)
        selector = torch.tensor(selector)
        new_selector = torch.zeros_like(selector)
        new_selector[indices] = selector
        new_edge_index = data.edge_index[:, new_selector]
        new_data = Data(x=data.x, edge_index=new_edge_index, y=data.y)
        new_data.train_mask = data.train_mask
        new_data.val_mask = data.val_mask
        new_data.test_mask = data.test_mask
        new_dataset = Dataset([new_data], num_features=dataset.num_features, num_classes=dataset.num_classes, task_type=dataset.task_type)
        torch.save(new_dataset, os.path.join(path, 'max_d_p' + str(drop_edge_portion).replace('.', '').replace('10', '1').replace('00', '0') + '.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparsification')
    parser.add_argument('--dataset_dir', default=None, type=str, help='Path to the directory containing preprocessed dataset.')
    args = parser.parse_args()

    empty(args.dataset_dir)
    rand(args.dataset_dir, [0.5, 0.9, 0.95, 0.99])
    max_d(args.dataset_dir, [0.5, 0.9, 0.95, 0.99])


