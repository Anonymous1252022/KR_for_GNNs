import datetime
import os
import subprocess
import tempfile
from typing import Any, Dict, Tuple, Union
import hydra as hydra
import pytorch_lightning as pl
import random

import string
import torch
import torch.utils.data
import torch_geometric.data
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data import GraphSAINTEdgeSampler, Data

from torch_geometric.transforms import BaseTransform
import pandas as pd


from graph_self_supervised_learning.dataset import get_dataset
from io import StringIO

from graph_self_supervised_learning.model_reg import ExpModelReg


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    print('The most free is GPU={} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx


def cfg2dict(cfg: DictConfig) -> Dict[str, Any]:
    """
    Recursively convert OmegaConf to vanilla dict
    """
    cfg_dict = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            cfg_dict[k] = cfg2dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict


class AddMaskTransform(BaseTransform):
    """
    Simple transform to add masks to data objects when using neighbour sampling
    """

    def __init__(self, target_mask: str = "root_nodes"):
        super().__init__()
        self.target_mask = target_mask

    def __call__(self, data: Any) -> Any:
        new_mask = torch.zeros_like(data.train_mask)
        new_mask[:data.batch_size] = True
        setattr(data, self.target_mask, new_mask)
        return data


def train_step(cfg: DictConfig,
               data: Union[Data, Tuple[Data, Data, Data]],
               model: ExpModelReg,
               logger):
    device_idx = int(cfg.enviroment.device.split("cuda:")[1])

    assert cfg.training.batch_size > 0
    if cfg.layer.name == "graphsage":
        num_neighbors = list(cfg.layer.num_neighbors)
    else:
        num_neighbors = [cfg.training.num_neighbours] * cfg.model.depth

    assert len(num_neighbors) == cfg.model.depth

    batch_size = cfg.training.batch_size
    if cfg.dataset.use_saint and cfg.layer.name != "graphsage":
        train_loader = GraphSAINTEdgeSampler(data=data,
                                             batch_size=batch_size,
                                             num_workers=cfg.enviroment.num_workers,
                                             prefetch_factor=5 if cfg.enviroment.num_workers > 0 else 2,
                                             pin_memory=True)
        val_loader = GraphSAINTEdgeSampler(data=data,
                                           batch_size=batch_size,
                                           num_workers=cfg.enviroment.num_workers,
                                           prefetch_factor=5 if cfg.enviroment.num_workers > 0 else 2,
                                           pin_memory=True)
        test_loader = GraphSAINTEdgeSampler(data=data,
                                            batch_size=batch_size,
                                            num_workers=cfg.enviroment.num_workers,
                                            prefetch_factor=5 if cfg.enviroment.num_workers > 0 else 2,
                                            pin_memory=True)
    else:
        train_loader = torch_geometric.loader.NeighborLoader(data=data,
                                                             num_neighbors=num_neighbors,
                                                             input_nodes=data.train_mask,
                                                             batch_size=batch_size,
                                                             num_workers=cfg.enviroment.num_workers,
                                                             prefetch_factor=5 if cfg.enviroment.num_workers > 0 else 2,
                                                             pin_memory=True,
                                                             transform=AddMaskTransform())
        val_loader = torch_geometric.loader.NeighborLoader(data=data,
                                                           num_neighbors=num_neighbors,
                                                           input_nodes=data.val_mask,
                                                           batch_size=batch_size,
                                                           num_workers=cfg.enviroment.num_workers,
                                                           prefetch_factor=5 if cfg.enviroment.num_workers > 0 else 2,
                                                           pin_memory=True,
                                                           transform=AddMaskTransform())
        test_loader = torch_geometric.loader.NeighborLoader(data=data,
                                                            num_neighbors=num_neighbors,
                                                            input_nodes=data.test_mask,
                                                            batch_size=batch_size,
                                                            num_workers=cfg.enviroment.num_workers,
                                                            transform=AddMaskTransform())


    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss/loss")

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=cfg.optimization.epochs,
        detect_anomaly=True,
        max_time=datetime.timedelta(hours=8),
        devices=[device_idx],
        accelerator="auto",
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval='step'),
                   EarlyStopping(monitor=f"val_loss/loss", min_delta=0.01,
                                 patience=cfg.optimization.early_stopping_tolerance, verbose=True, mode="min",
                                 strict=False),
                   checkpoint_callback,
                   ])
    trainer.fit(model, train_loader, (train_loader, val_loader, test_loader))
    try:
        # Get best
        best_ckpt_path = checkpoint_callback.best_model_path
        loaded_ckpt = torch.load(best_ckpt_path)
        model.load_state_dict(loaded_ckpt["state_dict"])
    except Exception: # If we did not complete one epoch
        trainer.validate(model=model, dataloaders=(train_loader, val_loader, test_loader)) # Always gurantee that we have result
    return model


def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True
    cfg.enviroment.device = f"cuda:{get_free_gpu()}"

    if cfg.enviroment.use_wandb:
        if cfg.enviroment.wandb_project_name == "":
            raise RuntimeError("You must give a project name when using logging results to wandb")
        wandb.init(project=cfg.enviroment.wandb_project_name, sync_tensorboard=True)
        wandb.config.update(cfg2dict(cfg))
        exp_name = wandb.run.name
    else:
        exp_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    cfg.enviroment.output_dir = os.path.join(get_original_cwd(), cfg.enviroment.output_dir, exp_name)
    cfg.enviroment.data_dir = os.path.join(get_original_cwd(), cfg.enviroment.data_dir)

    os.makedirs(cfg.enviroment.output_dir, exist_ok=True)
    print(f"Using output dir: {cfg.enviroment.output_dir}")

    with open(os.path.join(cfg.enviroment.output_dir, "config.yaml"), "w") as fp:
        OmegaConf.save(config=cfg, f=fp)

    data = get_dataset(cfg)

    logger = TensorBoardLogger(save_dir=cfg.enviroment.output_dir, log_graph=True)
    logger.log_hyperparams(cfg2dict(cfg))

    if cfg.dataset.multi_label_ds:
        if isinstance(data, Data):
            num_classes = data.y.size(1)
            num_features = data.x.size(1)
        else: # Tuple of 3 datasets
            assert len(data) == 3
            num_classes = data[0][0].y.size(1) # Suited for PPI
            num_features = data[0][0].x.size(1)
    else:
        num_classes = int(torch.max(data.y).item() + 1)
        num_features = data.x.size(1)
    model = ExpModelReg(cfg=cfg, in_channels=num_features, out_classes=num_classes)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("##########################################################")
        print(f"Training")
        print("##########################################################")
        model = train_step(cfg=cfg,
                           data=data,
                           model=model,
                           logger=logger)

        if cfg.enviroment.use_wandb and cfg.enviroment.upload_artifacts:
            # Save model artifact
            model_dst = os.path.join(tmp_dir, f"{exp_name}_model.pth")
            torch.save(model.state_dict(), model_dst)
            artifact = wandb.Artifact(f"{exp_name}_model", type="model")
            artifact.add_file(model_dst)
            wandb.run.log_artifact(artifact_or_path=artifact)

        wandb.finish()

@hydra.main(config_path="configs", config_name="default")  # Config name will be given via command line
def launcher(cfg: DictConfig):
    if cfg.is_empty():
        print("Nothing to do, no config given")
        return
    print(OmegaConf.to_yaml(cfg))
    try:
        main(cfg)
    except Exception as e:
        print(e)
        if cfg.enviroment.use_wandb:
            wandb.alert(
                title="Got exception while running",
                text=str(e)
            )
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    launcher()
