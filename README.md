# Recoverability
This repository contains the code to reproduce experiments from the paper: 
<div align="center">
    <h2>
        <b>
            Graph Representation Learning Through Recoverability
        </b>
    </h2>
</div>

where:
- `./synthetic_data` directory contains synthetic tests of recoverability loss estimation.
- `./graph_supervised_learning` directory contains the code which uses recoverability loss minimization along with supervised loss,
for training better node representations. For more details refer to [graph_supervised_learning/README.md](graph_supervised_learning/README.md).
- `./graph_self_supervised_learning` directory contains the code which uses **GIRL** (**G**raph **I**nformation **R**epresentation **L**earning) algorith
for self-supervised graph representation learning. The main idea behind algorithm depicted in image. For more details refer to [graph_self_supervised_learning/README.md](graph_self_supervised_learning/README.md).
![](images/GIRL.png?raw=true)


