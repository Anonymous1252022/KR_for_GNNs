## Self-Supervised Graph Representation Learning
### Overview
This directory holds the code used to reproduce the results we share in our paper under the node classification self supervision.  
You can add datasets and layers using the config folders, we encourage you to try different configuration and extend our work.

For more details on the training configuration check out our paper.
### Requirements
* `python==3.8.*`
* `requirements.txt`
* Weights and biases account is optional
* If you do not want to make use of faiss-gpu (not supported under some configurations), you can comment out the relevant code lines). The only outcome is that you want get mid training accuracy estimations. 
### Usage
* Each training session has a hard stop time limit, this is the time limit we used in our paper. You can modify the time limit in order to improve results.
  * Check out the trainer initialization code in `run_exp.py` and `run_exp_reg.py`.
#### Self-supervised
Just run the code in: `run_many_exp.py`, run the datasets and configurations you want.
#### Supervised
Just run the code in: `run_many_exp_reg.py`, run the datasets and configurations you want.
#### Fetch result
If you use weights and biases, you can fetch the results to a table using `fetch_results.py`, for more info call: `fetch_results.py --help`
