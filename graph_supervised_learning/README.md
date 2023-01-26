## Recoverabily in a Supervised Setting

### Overview
This directory contains the code which utilizes KR for better aggregation of information 
from neighboring nodes in GNN layers during training. The code contains dataset generation `datasets.py`,
KR loss estimator `rho.py` and `rho_estimator.py`, gnn training `train_gnn.py`, and the list of bash 
scripts with chosen hyperparameters located at `./scripts` directory.

For more details on the training configuration check out our paper.
### Requirements
* `python==3.8.*`
* `./requirements.txt`


### Steps to reproduce experiments:
- Create the empty directories outside the `KR_for_GNNs` repo according to the following structure:
```text
KR_for_GNNs/../
 ├── data/
 └── results/
      ├── Reddit/
      ├── Reddit2/
      ├── ogbn_arxiv/
      ├── PPI/
      ├── ogbn_products/
      └── rho_test/
```
- Run `datasets.py` script for generating datasets:
```bash
python ./datasets.py \
--output=../../data
```

- Run scripts in `./scripts` directory. Each script for a different dataset.
