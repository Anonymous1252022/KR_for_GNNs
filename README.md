# KR for GNNs
This repository contains the code to reproduce the experiments from the paper: 
<div align="center">
    <h2>
        <b>
            Graph Representation Learning via Aggregation Enhancement
        </b>
    </h2>
</div>

where:
- `./synthetic_data` directory contains synthetic tests of KR loss estimation.
- `./graph_supervised_learning` directory contains the code, where KR loss term is optimized along with supervised loss,
to obtain better node representations. For more details refer to [graph_supervised_learning/README.md](graph_supervised_learning/README.md).
- `./graph_self_supervised_learning` directory contains the code of **GIRL** (**G**raph **I**nformation **R**epresentation **L**earning) algorith
for self-supervised graph representation learning. The main idea behind algorithm depicted in the image. For more details refer to [graph_self_supervised_learning/README.md](graph_self_supervised_learning/README.md).
![](images/GIRL.png?raw=true)


