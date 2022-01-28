import os
import argparse
import torch
from torch_geometric.nn import GraphConv, SAGEConv, GINConv
from torch_geometric.data.data import Data
from dataset import Dataset
from torch_geometric.data import NeighborSampler
from tqdm import tqdm

class Embedding(torch.nn.Module):
    def __init__(self, num_features:int, type:str='GraphConv', eps:float=0.0):
        super(Embedding, self).__init__()
        if type in ['GraphConv', 'GCNConv']:
            self.layer = GraphConv(num_features, num_features, aggr='add')
        elif type == 'SAGEConv':
            self.layer = SAGEConv(num_features, num_features)
        elif type == 'GINConv':
            self.layer = GINConv(nn=lambda x: x, eps=eps)
        else:
            raise ValueError('Incorrect embedding type: ' + type)

        if type in ['GraphConv', 'GCNConv', 'SAGEConv']:
            parameters = list(self.layer.parameters())
            parameters[0].requires_grad = False
            parameters[0].data = torch.eye(num_features)
            parameters[1].requires_grad = False
            parameters[1].data.zero_()
            parameters[2].requires_grad = False
            parameters[2].data = torch.eye(num_features)

        if type == 'GCNConv':
            self.aux = GraphConv(1, 1, aggr='add')
            parameters = list(self.aux.parameters())
            parameters[0].requires_grad = False
            parameters[0].data = torch.eye(1)
            parameters[1].requires_grad = False
            parameters[1].data.zero_()
            parameters[2].requires_grad = False
            parameters[2].data = torch.eye(1)

        self.type = type

    def forward(self, x:torch.Tensor, orig_edge_index:torch.Tensor, sampler:NeighborSampler, device:torch.device, depth:int) -> torch.Tensor:
        if self.type == 'GCNConv':
            ones = torch.ones(x.shape[0], 1).to(device)
            norm_factor = (1. / torch.sqrt(self.aux(ones, orig_edge_index.to(device)))).cpu()
            x = x * norm_factor

        pbar = tqdm(total=x.shape[0])
        pbar.set_description(self.type + ' embedding depth=' + str(depth))
        xs = []
        for batch_size, n_id, adj in sampler:
            edge_index, original_edges, size = adj.to(device)
            x_ = x[n_id].to(device)
            x_target = x_[:size[1]]
            x_ = self.layer((x_, x_target), edge_index).cpu()

            if self.type == 'GCNConv':
                x_ = x_ * norm_factor[n_id][:size[1]]

            xs.append(x_)
            pbar.update(batch_size)
        x = torch.cat(xs, dim=0)
        pbar.close()
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding')
    parser.add_argument('--dataset', default=None, type=str, help='Path to pt file containing preprocessed dataset.')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size of the neighbor sampler.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of workers for neighbor sampler.')
    parser.add_argument('--num_embedding_layers', type=int, default=1)
    parser.add_argument('--dump_all_layers', type=bool, default=False, help='Whether to dump all embedding layers. If False, only the last embedding layer is dumped.')
    parser.add_argument('--embedding_type', default='GraphConv', type=str, help='{GraphConv, GCNConv, SAGEConv, GINConv}')
    parser.add_argument('--neighbor_sampler_size', default=-1, type=int, help='The size of neighbor sampler.')
    args = parser.parse_args()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = torch.load(args.dataset)
    data = dataset[0]

    # sampler
    sampler = NeighborSampler(data.edge_index, sizes=[args.neighbor_sampler_size], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, num_nodes=data.x.shape[0])

    # embedding
    embedding = Embedding(dataset.num_features, args.embedding_type, eps=10.0).to(device)

    x = data.x
    for i in range(args.num_embedding_layers):
        x = embedding(x, data.edge_index, sampler, device, i + 1)
        new_data = Data(edge_index=data.edge_index.T[[]].T, x=x, y=data.y)
        new_data.train_mask = data.train_mask
        new_data.val_mask = data.val_mask
        new_data.test_mask = data.test_mask
        new_dataset = Dataset([new_data], num_features=dataset.num_features, num_classes=dataset.num_classes, task_type=dataset.task_type)
        if args.dump_all_layers or (i+1==args.num_embedding_layers):
            torch.save(new_dataset, os.path.join(os.path.dirname(args.dataset), os.path.basename(args.dataset).split('.')[0] + '_' + args.embedding_type + '_depth_' + str(i + 1) + '.pt'))











