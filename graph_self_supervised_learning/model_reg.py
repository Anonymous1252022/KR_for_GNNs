from sklearn.metrics import f1_score

import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
import torch_geometric

from omegaconf import DictConfig


from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, GraphConv


class ExpModelReg(pl.LightningModule):

    def __init__(self, cfg: DictConfig, in_channels, out_classes):
        super().__init__()
        self.cfg = cfg
        self.out_classes = out_classes
        self.out_channels = out_classes

        last_layer_input_dim = cfg.model.hidden_dim if cfg.model.depth > 0 else in_channels
        self.last_layer = torch.nn.Sequential(torch.nn.Linear(last_layer_input_dim, cfg.model.hidden_dim),
                                              torch.nn.ELU(),
                                              torch.nn.Linear(cfg.model.hidden_dim, out_classes))

        self.conv_layers = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        if cfg.model.depth > 0:
            self.conv_layers.append(
                self.get_layer(in_channels=in_channels, out_channels=cfg.model.hidden_dim))
            if self.cfg.model.use_bn:
                self.bn.append(torch.nn.BatchNorm1d(cfg.model.hidden_dim))
        for i in range(cfg.model.depth - 1):
            self.conv_layers.append(
                self.get_layer(in_channels=cfg.model.hidden_dim, out_channels=cfg.model.hidden_dim))
            if self.cfg.model.use_bn:
                self.bn.append(torch.nn.BatchNorm1d(cfg.model.hidden_dim))

        self.activation = self.get_activation_func()
        self.criterion = None
        self.set_criterion()
        self._best_metrics_cache = {}

    def log(self, *args, **kwargs):
        print(f"{args[0]}: {args[1]:.2f}")
        super().log(*args, **kwargs)

    def get_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        layer_name = self.cfg.layer.name
        layer_params = self.cfg.layer.params
        if layer_name == "gcn":
            return GCNConv(in_channels=in_channels, out_channels=out_channels, **layer_params)
        elif layer_name == "gat":
            heads = layer_params.get("heads", 1)
            return GATConv(in_channels=in_channels, out_channels=out_channels // heads, **layer_params)
        elif layer_name == "gatv2":
            heads = layer_params.get("heads", 1)
            return GATv2Conv(in_channels=in_channels, out_channels=out_channels // heads, **layer_params)
        elif layer_name == "graphsage":
            return SAGEConv(in_channels=in_channels,
                            out_channels=out_channels,
                            **layer_params)
        elif layer_name == "graphconv":
            return GraphConv(in_channels=in_channels, out_channels=out_channels, **layer_params)
        else:
            raise RuntimeError(
                f"Invalid GNN layer got: {layer_name}")

    def get_activation_func(self) -> torch.nn.Module:
        activation_name = self.cfg.model.activation
        activation_map = {"elu": torch.nn.ELU(),
                          "leaky_relu": torch.nn.LeakyReLU(),
                          "relu": torch.nn.ReLU()}
        try:
            return activation_map[activation_name]
        except KeyError:
            raise RuntimeError(f"Invalid activation func, got: {activation_name}, available: {list(activation_map.keys())}")

    def _update_best_metric(self, metric, value, type: str, force: bool = False) -> bool:
        assert type in ("min", "max")
        updated_metric = False
        if metric in self._best_metrics_cache and not force:
            if type == "min" and value < self._best_metrics_cache[metric]:
                self._best_metrics_cache[metric] = value
                updated_metric = True
            if type == "max" and value > self._best_metrics_cache[metric]:
                self._best_metrics_cache[metric] = value
                updated_metric = True
        else:
            self._best_metrics_cache[metric] = value
            updated_metric = True

        self.log(f"best_{metric}", self._best_metrics_cache[metric])
        return updated_metric

    def configure_optimizers(self):
        optimization_cfg = self.cfg.optimization.optimizer
        ans = []
        optimizer = getattr(torch.optim, optimization_cfg.name)(self.parameters(),
                                                                         lr=optimization_cfg.initial_lr,
                                                                         weight_decay=optimization_cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",
                                                           factor=optimization_cfg.lr_sched_factor,
                                                           patience=optimization_cfg.lr_sched_patience,
                                                           verbose=True, min_lr=optimization_cfg.lr_sched_minimal_lr)

        return {"optimizer": optimizer, "lr_scheduler": sched, "monitor": "val_loss/loss"}

    def set_criterion(self):
        if self.cfg.dataset.multi_label_ds:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        self.criterion = criterion

    def forward(self, data: torch_geometric.data.Data):
        x = data.x
        edge_index = data.edge_index

        for i, curr_l in enumerate(self.conv_layers):
            x = curr_l(x, edge_index)

            if self.cfg.model.use_bn:
                x = self.bn[i](x)
            x = self.activation(x)

            if self.training:
                self.logger.experiment.add_histogram(f'embedding_histogram/conv_{i}', x.detach().cpu(),
                                                     self.global_step)

            if self.training and self.cfg.optimization.dropout_rate > 0:
                x = torch.nn.functional.dropout(x, p=self.cfg.optimization.dropout_rate)

        x = self.last_layer(x)

        return x

    def training_step(self, batch, batch_index):
        if hasattr(batch, "root_nodes"):
            criterion_mask = batch.root_nodes
        else:
            criterion_mask = batch.train_mask
        out = self.forward(batch)
        out = out[criterion_mask]
        y = batch.y[criterion_mask]
        loss = self.criterion(input=out, target=y)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if hasattr(batch, "root_nodes"):
            criterion_mask = batch.root_nodes
        else:
            criterion_mask = torch.ones_like(batch.train_mask)
        out = self.forward(batch)
        out = out[criterion_mask]
        y = batch.y[criterion_mask]
        loss = self.criterion(input=out, target=y)
        return out, y, loss

    def validation_epoch_end(self, validation_step_outputs):
        lvl_name = "final"
        conc_val_outputs = []
        for curr_outputs in validation_step_outputs:
            out, y, loss = zip(*curr_outputs)
            if len(out) == 1:
                out = out[0]
                y = y[0]
                loss = loss[0]
            else:
                out = torch.cat(out)
                y = torch.cat(y)
                loss = torch.stack(loss)
            conc_val_outputs.append((out.cpu().numpy(), y.cpu().numpy(), loss.cpu().numpy()))

        new_best_val = False

        for split_name, split_res in (("train", conc_val_outputs[0]), ("val", conc_val_outputs[1]), ("test", conc_val_outputs[2])):
            split_out, split_y, split_loss = split_res
            split_loss = np.mean(split_loss)
            self.log(f"{split_name}_loss/loss", split_loss)

            if self.cfg.dataset.multi_label_ds:
                preds = (split_out > 0).astype(np.int)
            else:
                preds = np.argmax(split_out, axis=-1)

            split_acc = np.mean(preds.flatten() == split_y.flatten()) * 100
            self.log(f"accuracy_{split_name}/lvl_{lvl_name}", split_acc)
            f1s = None
            if self.out_classes == 2 or self.cfg.dataset.multi_label_ds:
                # Compute F1 score
                f1s = f1_score(y_true=split_y, y_pred=np.reshape(preds, split_y.shape), average="micro")
                self.log(f"f1_score_{split_name}/lvl_{lvl_name}", f1s)

            self.logger.experiment.add_histogram(f'class_prediction_histogram/lvl_{lvl_name}', preds,
                                                 self.current_epoch)

            if split_name == "test" and new_best_val:
                # Assumption that test is running after val
                self._update_best_metric(metric=f"accuracy_{split_name}/lvl_{lvl_name}",
                                         value=split_acc,
                                         type="max",
                                         force=True)
                self._update_best_metric(metric=f"loss_{split_name}/lvl_{lvl_name}",
                                         value=split_loss,
                                         type="min",
                                         force=True)

                if f1s is not None:
                    self._update_best_metric(metric=f"f1_score_{split_name}/lvl_{lvl_name}",
                                             value=f1s,
                                             type="max",
                                             force=True)
            else:
                self._update_best_metric(metric=f"loss_{split_name}/lvl_{lvl_name}",
                                         value=split_loss,
                                         type="min")
                updated_acc = self._update_best_metric(metric=f"accuracy_{split_name}/lvl_{lvl_name}",
                                                       value=split_acc,
                                                       type="max")
                if split_name == "val" and updated_acc:
                    new_best_val = True
                    self.log(f"best_epoch/lvl_{lvl_name}", self.current_epoch)

                if f1s is not None:
                    self._update_best_metric(metric=f"f1_score_{split_name}/lvl_{lvl_name}",
                                             value=f1s,
                                             type="max")
