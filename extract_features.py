import os
import argparse
import torch
from torch_geometric.nn import SAGEConv, GATv2Conv
import torch.nn.functional as F
from torch_geometric.data import Data, NeighborSampler
import numpy as np
import random
from tqdm import tqdm
from dataset import Dataset
from typing import List

class GNN(torch.nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, num_embedding_layers:int, type:str='SAGEConv', heads:int=1):
        super(GNN, self).__init__()
        self.num_embedding_layers = num_embedding_layers
        self.embedding = torch.nn.ModuleList()

        if type == 'SAGEConv':
            if num_embedding_layers >= 1:
                self.embedding.append(SAGEConv(in_channels, hidden_channels))
                for _ in range(num_embedding_layers - 1):
                    self.embedding.append(SAGEConv(hidden_channels, hidden_channels))
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

    def inference(self, x_all:torch.Tensor, subgraph_loader:NeighborSampler, device:torch.device) -> List[torch.Tensor]:
        # Embedding part:
        pbar = tqdm(total=x_all.size(0) * self.num_embedding_layers)
        pbar.set_description('Feature extraction')
        features = []
        for i in range(self.num_embedding_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, original_edges, size = adj.to(device)
                x = x_all[n_id].to(device)

                x_target = x[:size[1]]
                x = self.embedding[i]((x, x_target), edge_index)
                x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)
            features.append(x_all)

        pbar.close()
        return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Features')
    parser.add_argument('--seed', default=-1, type=int, help='Seed >= 0.')
    parser.add_argument('--embedding_type', default='SAGEConv', type=str, help='{SAGEConv, GATv2Conv}')
    parser.add_argument('--heads', default=1, type=int, help='Number of heads in GATv2Conv.')
    parser.add_argument('--num_embedding_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dataset', default=None, type=str, help='Path to pt file containing preprocessed dataset. (.pt format)')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size of the neighbor sampler.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of workers for neighbor sampler.')
    parser.add_argument('--checkpoint_file_path', default=None, type=str, help='Path to load checkpoint file of last step. (.pth.tar format)')
    parser.add_argument('--neighbor_sampler_size', default=-1, type=int, help='The size of neighbor sampler.')
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

    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[args.neighbor_sampler_size], batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, num_nodes=data.x.shape[0])

    # model
    model = GNN(data.num_features, args.hidden_channels, dataset.num_classes, args.num_embedding_layers, args.embedding_type, args.heads)
    model.load_state_dict(torch.load(args.checkpoint_file_path))
    model = model.to(device)

    # extract features
    with torch.no_grad():
        model.eval()
        features = model.inference(data.x, subgraph_loader, device)
    for i in range(len(features)):
        x = features[i]
        new_data = Data(edge_index=data.edge_index.T[[]].T, x=x, y=data.y)
        new_data.train_mask = data.train_mask
        new_data.val_mask = data.val_mask
        new_data.test_mask = data.test_mask
        new_dataset = Dataset([new_data], num_features=dataset.num_features, num_classes=dataset.num_classes, task_type=dataset.task_type)
        torch.save(new_dataset, os.path.join(os.path.dirname(args.dataset), 'trained_' + args.embedding_type + '_features_layer_' + str(i + 1) + '.pt'))


