import os
import argparse
import torch
import numpy as np
import pandas as pd
from rho import compute_rho
from tqdm import tqdm
import random
from dataset import Dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute rho.')
    parser.add_argument('--dataset_dir', default=None, help='Path to the directory containing preprocessed datasets.')
    parser.add_argument('--batch_size', default=1000, type=int, help='Number of samples in rho estimations.')
    parser.add_argument('--lambda_', default=1.0, type=float, help='Kernel parameter.')
    parser.add_argument('--rho_log_file_path', default=None, type=str, help='Path to rho log file, for dump. (.csv format)')
    parser.add_argument('--seed', default=-1, type=int, help='Seed >= 0.')
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

    if args.rho_log_file_path is not None:
        columns = ['name', 'rho']
        pd.DataFrame(columns=columns).to_csv(args.rho_log_file_path, index=False)

    dataset_names = sorted(list(filter(lambda name: ('.pt' in name) and ('.pth.tar' not in name), os.listdir(args.dataset_dir))))
    for dataset_name in dataset_names:
        # dataset
        dataset = torch.load(os.path.join(args.dataset_dir, dataset_name))
        data = dataset[0]
        data.to(device)
        x = data.x[data.train_mask]
        y = data.y[data.train_mask]
        if dataset.task_type == 's':
            y = torch.nn.functional.one_hot(y.flatten()).type(torch.float32)
        else: # (m)
            y = y.type(torch.float32)

        steps = x.shape[0] // args.batch_size
        perm = torch.randperm(x.shape[0]).to(device)
        x = x[perm]
        y = y[perm]
        rho_list = []
        for idx in tqdm(range(steps)):
            try:
                rho_list.append(compute_rho(x[idx * args.batch_size: (idx + 1) * args.batch_size], y[idx * args.batch_size: (idx + 1) * args.batch_size], lambda_=args.lambda_))
            except RuntimeError:
                rho_list.append(np.nan)
        print('Estimated rho of ' + dataset_name + ': ' + str(np.array(rho_list).mean()))

        if args.rho_log_file_path is not None:
            pd.DataFrame(data=[[dataset_name, np.array(rho_list).mean()]], columns=columns).to_csv(
                args.rho_log_file_path, mode='a', index=False, header=False
            )



