## Recoverability Regularization

For training GNN model with recoverability regularization, use `train_gnn_reg.py` script.

- For `Reddit2` dataset run the following command:
```bash
python3 train_gnn_reg.py
--dataset=../data/Reddit2/ref.pt
--num_embedding_layers=9
--embedding_type=SAGEConv
--dropout=0.1
--lr=0.001
--epochs=20
--batch_size=1000
--neighbor_sampler_size=10
--seed=0
--train_log_file_path=../results/reg_test/Reddit2.csv
--rho_reg=1
--lambda_=1.0
````

- For `ogbn-arxiv`:
```bash
python3 train_gnn_reg.py
--dataset=../data/ogbn-arxiv/ref.pt
--num_embedding_layers=9
--embedding_type=SAGEConv
--dropout=0.1
--lr=0.001
--epochs=50
--batch_size=1000
--neighbor_sampler_size=10
--seed=0
--train_log_file_path=../results/reg_test/ogbn-arxiv.csv
--rho_reg=0.1
--lambda_=1.0
```

- For `ogbn-products`:
```bash
python3 train_gnn_reg.py
--dataset=../data/ogbn-products/ref.pt
--num_embedding_layers=9
--embedding_type=SAGEConv
--dropout=0.1
--lr=0.001
--epochs=20
--batch_size=1000
--neighbor_sampler_size=2
--seed=0
--train_log_file_path=../results/reg_test/ogbn-products.csv
--rho_reg=0.1
--lambda_=0.1
```

- For `Flickr`:
```bash
python3 train_gnn_reg.py
--dataset=../data/Flickr/ref.pt
--num_embedding_layers=9
--embedding_type=SAGEConv
--dropout=0.5
--lr=0.001
--epochs=20
--batch_size=1000
--neighbor_sampler_size=10
--seed=0
--train_log_file_path=../results/reg_test/Flickr.csv
--rho_reg=0.2
--lambda_=1.0
```

- And for `PPI`:
```bash
python train_gnn_reg.py
--dataset=../data/PPI/ref.pt
--num_embedding_layers=9
--embedding_type=SAGEConv
--dropout=0.1
--lr=0.001
--epochs=50
--batch_size=1000
--neighbor_sampler_size=10
--seed=0
--train_log_file_path=../results/reg_test/PPI.csv
--rho_reg=0.01
--lambda_=0.7
```
