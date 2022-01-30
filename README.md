# Recoverability
This repo contains the code to reproduce experiments from the paper: 
<div align="center">
    <h2>
        <b>
            On Recoverability of Graph Neural Network Representation
        </b>
    </h2>
</div>

The main idea behind the recoverability is visualized in the following figure, 
where we can see how recoverability loss rho changes when the input graph is sparsified.
When the graph is sparser the recoverability loss is higher. We can use this effect to 
understand why some sparsification methods are better the others, and potentially propose
better one in the future.
![](images/gs_vs_rho.png?raw=true)

## Requirements:
- python3.6
- numpy==1.18.1
- pandas==1.0.1
- tqdm==4.43.0
- matplotlib==3.3.4
- scikit-learn==0.23.2
- torch==1.9.0
- torch-scatter==2.0.8
- torch-sparse==0.6.11
- torch-cluster==1.5.9
- torch-spline-conv==1.2.1
- torch-geometric==1.7.2
- ogb==1.3.2

## Steps for the reproduction of the experiments:
- Create the empty directories inside the `Recoverability` repo according to the following structure:
```text
Recoverability/
 ├── data/
 └── results/
      ├── Reddit2/
      ├── ogbn-arxiv/
      ├── Flickr/
      ├── PPI/
      ├── ogbn-products/
      └── rho_test/
```

- Run `test_rho.py` script to test the effectiveness of the recoverability loss metric.
- For dataset preparation use `prepare_datasets.py` script.
It first, downloads five datasets into the `data` directory, then does preprocessing
and saves the results at `ref.pt` files.

- For dataset sprsification use `sparsify_datasets.py` script.
```bash
python sparsify_datasets.py
--dataset_dir=./data/<dataset>
```
where the `<dataset>` is one of the following `Reddit2, ogbn-arxiv, Flickr, PPI, ogbn-products`.
The script applies two sparsification methods, <b>Random</b> and <b>Max d</b>,  to the `ref.pt` file, and saves the sparsified
versions into `./data/<dataset>` directory. For instance the 90% drop of the edges according <b>Max d</b> method will be saved
at `max_d_p09.pt` file.

- For training GNN model use `train_gnn.py` script.
```bash
python train_gnn.py
--dataset=./data/<dataset>/ref.pt
--train_log_file_path=./results/<dataset>/ref_<embedding type>_depth_<depth>.csv
--num_embedding_layers=<depth>
--hidden_channels=<hidden channels>
--embedding_type=<embedding type>
--dropout=<dropout>
--seed=0
--lr=0.001
--epochs=<epochs>
--batch_size=1000
--neighbor_sampler_size=10
--save_checkpoint_file_path=./data/<dataset>/<embedding type>_depth_<depth>_weights.pth.tar
```
where the `<dataset>` is one of the following `Reddit2, ogbn-arxiv, Flickr, PPI, ogbn-products`,
the `<depth>` is one of the following `1,2,3,4,5,6,7,8,9`.
and the `<embedding type>` is one of the following `GraphConv, GCNConv, SAGEConv, GINConv, GATv2Conv`.
For `Flickr` dataset 0.5 dropout is used, and for the rest, 0.1
The `ogbn-arxiv` and `PPI` datasets are trained 50 epochs, and the rest, 20 epochs.
The training log is saved in .csv file `ref_<embedding type>_depth_<depth>.csv`.

- For appling embedding (only the aggregation part) use `apply_embedding.py` script.
```bash
python apply_embedding.py
--dataset=./data/<dataset>/ref.pt
--embedding_type=<embedding type>
--num_embedding_layers=<depth>
--neighbor_sampler_size=-1
--dump_all_layers=True
```
where the `<dataset>` is one of the following `Reddit2, ogbn-arxiv, Flickr, PPI, ogbn-products`,
and the `<embedding type>` is one of the following `GraphConv, GCNConv, SAGEConv, GINConv`.
The script saves .pt files inside `./data/<dataset>` with the following name template `ref_<embedding type>_depth_<depth>.pt`.

- For features extraction of the trained GNN model use `extract_features.py` script.
```bash
python extract_features.py
--seed=0
--embedding_type=<embedding type>
--num_embedding_layers=<depth>
--dataset=./data/PPI/ref.pt
--checkpoint_file_path=./data/<dataset>/<embedding type>_depth_<depth>_weights.pth.tar
--neighbor_sampler_size=-1
```
where the `<dataset>` is one of the following `Reddit2, ogbn-arxiv, Flickr, PPI, ogbn-products`,
the `<depth>` is one of the following `1,2,3,4,5,6,7,8,9`.
and the `<embedding type>` is one of the following `SAGEConv, GATv2Conv`.
The script saves .pt files inside `./data/<dataset>` with the following name template `trained_<embedding type>_features_layer_<depth>.pt`.

- To compute rho for all embeddings use `compute_rho.py` script.
```bash
python compute_rho.py
--dataset_dir=./data/<dataset>
--lambda_=<lambda>
--rho_log_file_path=./results/<dataset>/rho.csv
--seed=0
```
where the `<dataset>` is one of the following `Reddit2, ogbn-arxiv, Flickr, PPI, ogbn-products`.
For `PPI` dataset use `--lambda_=0.5`, for `ogbn-products` use `--lambda_=0.1` and for the rest datasets use `--lambda_=1.0`.
The results are saved in .csv file `rho.csv`.



