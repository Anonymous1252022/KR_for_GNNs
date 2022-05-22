import os
import argparse
import torch
from torch_geometric.nn import GraphConv, SAGEConv, GINConv, GATv2Conv
import torch.nn.functional as F
from torch_geometric.data import Data, NeighborSampler
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from typing import Union, List
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from dataset import Dataset
from regularization.rho_estimator import RhoEstimator
del sys.path[0]

class GNN(torch.nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, num_embedding_layers:int, dropout:float,
                 type:str='GraphConv', heads:int=1,  eps:float=10.0, norm_factor:Union[torch.Tensor, type(None)]=None, task_type:str='s'):
        super(GNN, self).__init__()
        self.num_embedding_layers = num_embedding_layers
        self.embedding = torch.nn.ModuleList()
        if (type == 'GraphConv') or (type == 'GCNConv'):
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
        self.norm_factor = norm_factor
        self.dropout = dropout
        assert task_type in ['s', 'm'], "task_type should be one of {'s', 'm'}"
        self.task_type = task_type

    def forward(self, x_all:torch.Tensor, n_id:torch.Tensor, adjs:torch.Tensor) -> torch.Tensor:
        # Embedding part:
        x = x_all[n_id]
        if self.type == 'GCNConv':
            norm_factor = self.norm_factor[n_id]
            x = x * norm_factor

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.embedding[i]((x, x_target), edge_index)

            if self.type == 'GCNConv':
                norm_factor = norm_factor[:size[1]]
                x = x * norm_factor

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

    def inference(self, x_all:torch.Tensor, subgraph_loader:NeighborSampler, device:torch.device) -> torch.Tensor:
        # Embedding part:
        pbar = tqdm(total=x_all.size(0) * self.num_embedding_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_embedding_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, original_edges, size = adj.to(device)
                x = x_all[n_id].to(device)

                if self.type == 'GCNConv':
                    norm_factor = self.norm_factor[n_id]
                    x = x * norm_factor

                x_target = x[:size[1]]
                x = self.embedding[i]((x, x_target), edge_index)

                if self.type == 'GCNConv':
                    norm_factor = norm_factor[:size[1]]
                    x = x * norm_factor

                x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        # Classifier part:
        x = x_all.to(device)
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)

            if i != len(self.classifier) - 1:
                x = F.relu(x)

        x_all = x.cpu()

        return x_all


def train(model:GNN, rho_estimator:RhoEstimator, optimizer:torch.optim.Adam, data:Data, train_loader:NeighborSampler, epoch:int, device:torch.device) -> float:
    model.train()
    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')
    x = data.x.to(device)
    y = data.y.to(device)
    total_loss = 0

    for batch_size, n_id, adjs in train_loader:
        if not isinstance(adjs, list):
            adjs = [adjs]
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()

        rho_estimator.set_y(y[n_id[:batch_size]])
        rho_estimator.set_batch_size(batch_size)

        out = model(x, n_id, adjs)
        if model.task_type == 's':
            loss = F.nll_loss(out, y[n_id[:batch_size]])
        else: # (m)
            loss = F.binary_cross_entropy_with_logits(out, y[n_id[:batch_size]])

        rho = rho_estimator.get_rho()
        loss = loss + rho
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(model:GNN, data:Data, subgraph_loader:NeighborSampler, device:torch.device) -> List[float]:
    model.eval()
    x = data.x
    y = data.y

    out = model.inference(x, subgraph_loader, device)

    results = []
    if model.task_type == 's':
        y_true = y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    else: # (m)
        y_true = y.cpu()
        y_pred = (out > 0).float().cpu()
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            results += [f1_score(y_true[mask], y_pred[mask], average='micro')]

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', default=-1, type=int, help='Seed >= 0.')
    parser.add_argument('--embedding_type', default='GraphConv', type=str, help='{GraphConv, GCNConv, SAGEConv, GINConv, GATv2Conv}')
    parser.add_argument('--heads', default=1, type=int, help='Number of heads in GATv2Conv.')
    parser.add_argument('--num_embedding_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dataset', default=None, type=str, help='Path to pt file containing preprocessed dataset. (.pt format)')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size of the neighbor sampler.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of workers for neighbor sampler.')
    parser.add_argument('--neighbor_sampler_size', default=10, type=int, help='The size of neighbor sampler.')
    parser.add_argument('--train_log_file_path', default=None, type=str, help='Path to train log file, for dump. (.csv format)')
    parser.add_argument('--save_checkpoint_file_path', default=None, type=str, help='Path to save checkpoint file of last step. (.pth.tar format)')
    parser.add_argument('--do_only_eval_from_ckpt', default=None, type=str, help='If given path to checkpoint, will do only evaluation.')
    parser.add_argument('--dump_rho', default=None, type=str, help='Path to dump rho.')
    parser.add_argument('--rho_reg', default=0.0, type=float, help='Rho regularization parameter.')
    parser.add_argument('--lambda_', default=1.0, type=float, help='Kernel parameter.')

    args = parser.parse_args()

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

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=args.num_embedding_layers * [args.neighbor_sampler_size], batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, num_nodes=data.x.shape[0])
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, num_nodes=data.x.shape[0])

    # model
    norm_factor = None
    if args.embedding_type == 'GCNConv':
        aux = GraphConv(1, 1, aggr='add')
        parameters = list(aux.parameters())
        parameters[0].requires_grad = False
        parameters[0].data = torch.eye(1)
        parameters[1].requires_grad = False
        parameters[1].data.zero_()
        parameters[2].requires_grad = False
        parameters[2].data = torch.eye(1)
        aux = aux.to(device)
        ones = torch.ones(data.x.shape[0], 1).to(device)
        norm_factor = (1. / torch.sqrt(aux(ones, data.edge_index.to(device))))

    model = GNN(data.num_features, args.hidden_channels, dataset.num_classes, args.num_embedding_layers, args.dropout, args.embedding_type, heads=args.heads, norm_factor=norm_factor, task_type=dataset.task_type).to(device)
    rho_estimator = RhoEstimator(path=args.dump_rho, model=model, layers_type=[SAGEConv], rho_reg=args.rho_reg, lambda_=args.lambda_, task_type=dataset.task_type)

    if args.do_only_eval_from_ckpt is not None:
        model.load_state_dict(torch.load(args.do_only_eval_from_ckpt))
        train_acc, val_acc, test_acc = test(model, data, subgraph_loader, device)
        print(f'train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')
        exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.train_log_file_path is not None:
        columns = ['epoch', 'loss', 'train_acc', 'val_acc', 'test_acc']
        pd.DataFrame(columns=columns).to_csv(args.train_log_file_path, index=False)

    for epoch in range(args.epochs):
        loss = train(model, rho_estimator, optimizer, data, train_loader, epoch, device)
        train_acc, val_acc, test_acc = test(model, data, subgraph_loader, device)
        print(f'epoch {epoch:02d}, loss: {loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')

        if args.train_log_file_path is not None:
            pd.DataFrame(data=[[epoch, loss, train_acc, val_acc, test_acc]], columns=columns).to_csv(
                args.train_log_file_path, mode='a', index=False, header=False
            )

    if args.save_checkpoint_file_path is not None:
        torch.save(model.to('cpu').state_dict(), os.path.join(args.save_checkpoint_file_path))