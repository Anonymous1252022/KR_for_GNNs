import os
import argparse
import torch
from torch_geometric.nn import GraphConv, SAGEConv, GINConv, GATv2Conv
import torch.nn.functional as F
from torch_geometric.data import Data, NeighborSampler, ClusterData, ClusterLoader
import numpy as np
import random
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Union, List, Tuple
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from graph_supervised_learning.rho_estimator import RhoEstimator
from graph_supervised_learning.datasets import Dataset
del sys.path[0]

def save_data(data_, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data_, f)

def load_data(fname):
    with open(fname, 'rb') as f:
        data_ = pickle.load(f)
    return data_

class GNN(torch.nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, num_embedding_layers:int, dropout:float,
                 type:str='GraphConv', heads:int=1,  eps:float=10.0, task_type:str='s'):
        super(GNN, self).__init__()
        self.num_embedding_layers = num_embedding_layers
        self.embedding = torch.nn.ModuleList()
        if type == 'GraphConv':
            if num_embedding_layers >= 1:
                self.embedding.append(GraphConv(in_channels, hidden_channels, aggr='add'))
                for _ in range(num_embedding_layers - 1):
                    self.embedding.append(GraphConv(hidden_channels, hidden_channels, aggr='add'))
        elif type == 'SAGEConv':
            if num_embedding_layers >= 1:
                self.embedding.append(SAGEConv(in_channels, hidden_channels))
                for _ in range(num_embedding_layers - 1):
                    self.embedding.append(SAGEConv(hidden_channels, hidden_channels))
        elif type == 'GINConv':
            if num_embedding_layers >= 1:
                self.embedding.append(GINConv(nn=torch.nn.Linear(in_channels, hidden_channels), eps=eps))
                for _ in range(num_embedding_layers - 1):
                    self.embedding.append(GINConv(nn=torch.nn.Linear(hidden_channels, hidden_channels), eps=eps))
        elif type == 'GATv2Conv':
            if num_embedding_layers >= 1:
                self.embedding.append(GATv2Conv(in_channels, hidden_channels//heads, heads=heads))
                for _ in range(num_embedding_layers - 1):
                    self.embedding.append(GATv2Conv(hidden_channels, hidden_channels//heads, heads=heads))
        else:
            raise ValueError('Incorrect embedding type: ' + type)

        self.classifier = torch.nn.ModuleList()
        if num_embedding_layers <= 0:
            self.classifier.append(torch.nn.Linear(in_channels, hidden_channels))
            self.classifier.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.classifier.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.classifier.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.classifier.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.classifier.append(torch.nn.Linear(hidden_channels, out_channels))

        self.type = type
        self.dropout = dropout
        assert task_type in ['s', 'm'], "task_type should be one of {'s', 'm'}"
        self.task_type = task_type

    def forward(self, x:torch.Tensor, edge_index:torch.Tensor) -> torch.Tensor:
        # Embedding part:
        for i in range(len(self.embedding)):
            x = self.embedding[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Classifier part:
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)

            if i != len(self.classifier) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task_type == 's':
            return x.log_softmax(dim=-1)
        else: # (m)
            return x


def train(model:GNN, rho_estimator:RhoEstimator, optimizer:torch.optim.Adam, data:Data, data_loader:ClusterLoader, epoch:int, device:torch.device) -> Tuple[float, float, float, float, float]:
    # train
    model.train()
    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Train Epoch {epoch:02d}')
    loss_acc = rho_acc = 0
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if rho_estimator is not None:
            if rho_estimator.rho_reg != 0:
                rho_estimator.set_mask(batch.train_mask)
                rho_estimator.set_y(batch.y)

        out = model(batch.x, batch.edge_index)
        if model.task_type == 's':
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        else: # (m)
            loss = F.binary_cross_entropy_with_logits(out[batch.train_mask], batch.y[batch.train_mask])

        total_loss = loss
        if rho_estimator is not None:
            if rho_estimator.rho_reg != 0:
                total_loss = loss + rho_estimator.rho_reg * rho_estimator.rho
        total_loss.backward()
        optimizer.step()

        loss_acc += float(loss)
        if rho_estimator is not None:
            if rho_estimator.rho_reg != 0:
                rho_acc += float(rho_estimator.rho)
        pbar.update(int(batch.train_mask.sum()))

    pbar.close()
    loss = loss_acc / len(data_loader)
    rho = rho_acc / len(data_loader)

    # eval
    with torch.no_grad():
        model.eval()
        pbar = tqdm(total=int(data.num_nodes))
        pbar.set_description(f'Eval Epoch {epoch:02d}')
        train_acc = val_acc = test_acc = 0
        train_y_true_acc = []
        train_y_pred_acc = []
        val_y_true_acc = []
        val_y_pred_acc = []
        test_y_true_acc = []
        test_y_pred_acc = []
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            if model.task_type == 's':
                y_true = batch.y.unsqueeze(-1)
                y_pred = out.argmax(dim=-1, keepdim=True)
                train_acc += int(y_pred[batch.train_mask].eq(y_true[batch.train_mask]).sum())
                val_acc += int(y_pred[batch.val_mask].eq(y_true[batch.val_mask]).sum())
                test_acc += int(y_pred[batch.test_mask].eq(y_true[batch.test_mask]).sum())
            else:  # (m)
                y_true = batch.y.cpu()
                y_pred = (out > 0).float().cpu()
                train_y_true_acc.append(y_true[batch.train_mask])
                train_y_pred_acc.append(y_pred[batch.train_mask])
                val_y_true_acc.append(y_true[batch.val_mask])
                val_y_pred_acc.append(y_pred[batch.val_mask])
                test_y_true_acc.append(y_true[batch.test_mask])
                test_y_pred_acc.append(y_pred[batch.test_mask])

            pbar.update(int(batch.num_nodes))

        pbar.close()
        if model.task_type == 's':
            results = [train_acc / int(data.train_mask.sum()), val_acc / int(data.val_mask.sum()), test_acc / int(data.test_mask.sum())]
        else: # (m)
            train_y_true = torch.cat(train_y_true_acc).flatten()
            train_y_pred = torch.cat(train_y_pred_acc).flatten()
            val_y_true = torch.cat(val_y_true_acc).flatten()
            val_y_pred = torch.cat(val_y_pred_acc).flatten()
            test_y_true = torch.cat(test_y_true_acc).flatten()
            test_y_pred = torch.cat(test_y_pred_acc).flatten()
            train_acc += f1_score(train_y_true, train_y_pred, average='micro')
            val_acc += f1_score(val_y_true, val_y_pred, average='micro')
            test_acc += f1_score(test_y_true, test_y_pred, average='micro')
            results = [train_acc, val_acc, test_acc]

    return loss, rho, results[0], results[1], results[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train gnn')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', default=-1, type=int, help='Seed >= 0.')
    parser.add_argument('--embedding_type', default='GraphConv', type=str, help='{GraphConv, SAGEConv, GINConv, GATv2Conv}')
    parser.add_argument('--heads', default=1, type=int, help='Number of heads in GATv2Conv.')
    parser.add_argument('--num_embedding_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dataset', default=None, type=str, help='Path to pt file containing preprocessed dataset. (.pt format)')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size of the neighbor sampler.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for neighbor sampler.') # TODO: default it was 12, there is a bug with num_workers>0
    parser.add_argument('--num_clusters', default=1500, type=int, help='Number of clusters METIS partitioning.')
    parser.add_argument('--output_dir', default=None, type=str, help='Path to output directory.')
    parser.add_argument('--rho_reg', default=0.0, type=float, help='Rho regularization parameter.')
    parser.add_argument('--lambda_', default=1.0, type=float, help='Kernel sensitivity parameter.')

    args = parser.parse_args()
    if args.output_dir is not None:
        save_data(args, os.path.join(args.output_dir, 'args.pkl'))

    # seed
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = torch.load(args.dataset)
    data = dataset[0]
    data.to(device)
    cluster_data = ClusterData(data, num_parts=args.num_clusters, recursive=False, save_dir=os.path.dirname(args.dataset))
    data_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # model
    model = GNN(data.num_features, args.hidden_channels, dataset.num_classes, args.num_embedding_layers, args.dropout, args.embedding_type, heads=args.heads, task_type=dataset.task_type).to(device)
    if args.rho_reg != 0:
        if args.output_dir is not None:
            rho_dump_file = os.path.join(args.output_dir, 'rho.csv')
        else:
            rho_dump_file = None

        embedding_type_dict = {'GraphConv': GraphConv, 'SAGEConv': SAGEConv, 'GINConv': GINConv, 'GATv2Conv': GATv2Conv}
        rho_estimator = RhoEstimator(path=rho_dump_file, model=model, layers_type=[embedding_type_dict[args.embedding_type]], rho_reg=args.rho_reg, lambda_=args.lambda_, task_type=dataset.task_type)
    else:
        rho_estimator = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.output_dir is not None:
        columns = ['epoch', 'loss', 'rho', 'train_acc', 'val_acc', 'test_acc']
        pd.DataFrame(columns=columns).to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)

    for epoch in range(args.epochs):
        loss, rho, train_acc, val_acc, test_acc = train(model, rho_estimator, optimizer, data, data_loader, epoch, device)
        print(f'epoch {epoch:02d}, loss: {loss:.4f}, rho: {rho:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')

        if args.output_dir is not None:
            pd.DataFrame(data=[[epoch, loss, rho, train_acc, val_acc, test_acc]], columns=columns).to_csv(
                os.path.join(args.output_dir, 'train.csv'), mode='a', index=False, header=False
            )
