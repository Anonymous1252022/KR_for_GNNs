import enum
import itertools
from typing import List, Optional, Tuple
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
import torch_geometric
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omegaconf import DictConfig
from sklearn.metrics import RocCurveDisplay, auc, roc_auc_score, roc_curve, confusion_matrix

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, GraphConv

from unsupervised.gaussian_kernel import GaussianKernel
from unsupervised.knn import KNNFaiss


class EModelPhase(enum.Enum):
    TRAIN_EMBEDDING = "train_embedding"
    TRAIN_CLASSIFIER = "train_classifier"

class ExpModel(pl.LightningModule):

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
        self.is_last_step = None
        self.criterion = None
        self.phase = None
        self._best_metrics_cache = {}

        self.automatic_optimization = False

    def log(self, *args, **kwargs):
        print(f"{args[0]}: {args[1]:.2f}")
        super().log(*args, **kwargs)

    def set_phase(self, phase: EModelPhase) -> None:
        self.phase = phase
        self.set_criterion()

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
        if self.phase == EModelPhase.TRAIN_EMBEDDING:  # All except the classifier
            parameters = []
            for i in range(len(self.conv_layers)):
                curr_params = [self.conv_layers[i].parameters()]
                if self.cfg.model.use_bn:
                    curr_params.append(self.bn[i].parameters())

                parameters.append(itertools.chain(*curr_params))
        elif self.phase == EModelPhase.TRAIN_CLASSIFIER:  # Only the classifier
            parameters = [self.last_layer.parameters()]
        else:
            raise RuntimeError(f"Invalid model phase: {self.phase}")

        optimization_cfg = self.cfg.optimization.optimizer
        ans = []
        for curr_params in parameters:
            curr_params = [p for p in curr_params]
            optimizer = getattr(torch.optim, optimization_cfg.name)(curr_params,
                                                                             lr=optimization_cfg.initial_lr,
                                                                             weight_decay=optimization_cfg.weight_decay)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",
                                                               factor=optimization_cfg.lr_sched_factor,
                                                               patience=optimization_cfg.lr_sched_patience,
                                                               verbose=True, min_lr=optimization_cfg.lr_sched_minimal_lr)

            ans.append({"optimizer": optimizer, "lr_scheduler": sched, "monitor": "val_loss/loss"})

        return ans

    def set_criterion(self):
        unsupervised_training = self.phase == EModelPhase.TRAIN_EMBEDDING

        if unsupervised_training:
            kernel = GaussianKernel(max_samples=4096, add_regularization=self.cfg.training.add_regularization)
            criterion = lambda x, y: kernel.compute_d(x, y)
        else:
            if self.cfg.dataset.multi_label_ds:
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = torch.nn.CrossEntropyLoss()

        self.criterion = criterion

    def forward(self, data: torch_geometric.data.Data, criterion_mask: torch.Tensor, compute_loss: bool = True):
        y = data.y
        edge_index = data.edge_index

        out = self._internal_forward(data)

        lvl_loss = global_loss = None

        if compute_loss:
            if self.phase == EModelPhase.TRAIN_EMBEDDING:  # We take only the edges which connect to the root nodes
                root_nodes_idx = torch.where(criterion_mask)[0]
                relevant_edges = sum(data.edge_index[1] == i for i in root_nodes_idx).bool()
                if len(relevant_edges) == 0:
                    print("Warning got graph without edges in embedding phase")
                    return None, None, out, y
                relevant_nodes = torch.cat([data.edge_index[0][relevant_edges], root_nodes_idx], dim=0)
                relevant_nodes = torch.unique(relevant_nodes)
                edge_index = torch_geometric.utils.subgraph(relevant_nodes, data.edge_index, num_nodes=data.x.size(0))[0]
                try:
                    lvl_loss, global_loss = self._calculate_loss(edge_index=edge_index, out=out, y=y)
                except RuntimeError as re:
                    print(f"Got runtime error when calculating loss, probably SVD problem: {re}")
                    return None, None, out, y
                out = [o[criterion_mask] for o in out] # Just to keep the function consistent
                y = data.y[criterion_mask]
            else:
                out = [o[criterion_mask] for o in out]
                y = data.y[criterion_mask]
                lvl_loss, global_loss = self._calculate_loss(edge_index=edge_index, out=out, y=y)
        return lvl_loss, global_loss, out, y

    def _internal_forward(self, data: torch_geometric.data.Data):
        add_stop_gradients_lvl = self.phase == EModelPhase.TRAIN_EMBEDDING

        x = data.x
        edge_index = data.edge_index

        out = [x]
        for i, curr_l in enumerate(self.conv_layers):
            x = curr_l(x, edge_index)

            if self.cfg.model.use_bn:
                x = self.bn[i](x)
            x = self.activation(x)

            if self.training:
                self.logger.experiment.add_histogram(f'embedding_histogram/conv_{i}', x.detach().cpu(),
                                                     self.global_step)

            out.append(x)

            if add_stop_gradients_lvl and self.training:
                x = x.clone()
                x.register_hook(lambda grad: torch.zeros_like(grad))

            if self.training and self.cfg.optimization.dropout_rate > 0:
                x = torch.nn.functional.dropout(x, p=self.cfg.optimization.dropout_rate)

        x = self.last_layer(x)
        out.append(x)

        return out

    def _calculate_loss(self, edge_index: torch.Tensor, out: List[torch.Tensor], y: Optional[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        loss = [None] * len(out)
        if self.phase == EModelPhase.TRAIN_EMBEDDING:
            global_loss = 0
            for i in range(1, len(out) - 1):  # We don't care about the classifier and the first embedding
                relevant_edges = edge_index.T
                if relevant_edges.shape[0] == 0:
                    loss[i] = None
                    continue
                elif relevant_edges.shape[0] > self.cfg.training.max_edges_for_loss:
                    idx_to_take = torch.randperm(relevant_edges.shape[0])[:self.cfg.training.max_edges_for_loss]
                    relevant_edges = relevant_edges[idx_to_take]
                neighbours_emb = out[i - 1]  # We want to be able to reproduce the neighbours from the agg nodes
                target_emb = out[i]
                source_nodes, target_nodes = torch.split(relevant_edges, 1, dim=1)
                source_nodes = source_nodes.flatten()
                target_nodes = target_nodes.flatten()

                # Need to detach the neighbours from the loss calculation
                neighbours_emb = neighbours_emb.clone()

                if neighbours_emb.requires_grad:
                    neighbours_emb.register_hook(lambda grad: torch.zeros_like(grad))
                selected_neighbours = neighbours_emb[source_nodes]

                selected_targets = target_emb[target_nodes]

                lvl_loss = self.criterion(x=selected_targets, y=selected_neighbours)
                if self.cfg.training.use_self_in_loss:
                    lvl_loss += self.criterion(x=selected_targets, y=neighbours_emb[target_nodes])
                if torch.any(torch.isnan(lvl_loss)).item():
                    print("Warning: got nan in loss computation")
                    loss[i] = None
                    continue

                loss[i] = lvl_loss
                global_loss += lvl_loss
        else:
            last_layer_loss = self.criterion(out[-1], y)
            loss[-1] = last_layer_loss
            global_loss = last_layer_loss
        return loss, global_loss

    def forward_single_conv(self, layer_idx: int, x: torch.Tensor, edge_index:torch.Tensor) -> torch.Tensor:
        x = self.conv_layers[layer_idx](x, edge_index)
        if self.cfg.model.use_bn:
            x = self.bn[layer_idx](x)
        x = self.activation(x)
        return x

    def forward_only_classifier(self, x: torch.Tensor, y: torch.Tensor, compute_loss: bool=True):
        if self.training and self.cfg.optimization.dropout_rate > 0:
            x = torch.nn.functional.dropout(x, p=self.cfg.optimization.dropout_rate)
        x = self.last_layer(x)
        loss = None
        if compute_loss:
            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                y = y.type(torch.float32)
            loss = self.criterion(x, y)
        return [loss], loss, [x], y

    def training_step(self, batch, batch_index):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        if self.phase == EModelPhase.TRAIN_EMBEDDING or (not self.cfg.dataset.pre_calculate_embeddings):
            if hasattr(batch, "root_nodes"):
                criterion_mask = batch.root_nodes
            else:
                criterion_mask = torch.ones_like(batch.train_mask)
            lvl_loss, global_loss, out, y = self.forward(data=batch, criterion_mask=criterion_mask)
        elif self.phase == EModelPhase.TRAIN_CLASSIFIER:
            lvl_loss, global_loss, out, y = self.forward_only_classifier(x=batch[0], y=batch[1])
        else:
            raise RuntimeError("Invalid phase")

        [o.zero_grad() for o in optimizers]
        if isinstance(global_loss, torch.Tensor):  # If no edges, this will be an int if we are in embedding training
            self.manual_backward(global_loss, retain_graph=True)
            [o.step() for o in optimizers]
            self.log("loss_train/global", global_loss.cpu().item())
            if self.phase == EModelPhase.TRAIN_EMBEDDING:
                for i, curr_lvl_loss in enumerate(lvl_loss):
                    if curr_lvl_loss is not None:
                        self.log(f"loss_train/lvl_{i}", curr_lvl_loss.cpu().item())
            else:
                self.log(f"loss_train/final", global_loss.cpu().item())

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.phase == EModelPhase.TRAIN_CLASSIFIER and self.cfg.dataset.pre_calculate_embeddings:
            lvl_loss, global_loss, out, y = self.forward_only_classifier(x=batch[0], y=batch[1])
            return out, y, lvl_loss, global_loss
        else:
            if hasattr(batch, "root_nodes"):
                criterion_mask = batch.root_nodes
            else:
                criterion_mask = torch.ones_like(batch.train_mask)
            lvl_loss, global_loss, out, y = self.forward(data=batch, criterion_mask=criterion_mask, compute_loss=True)
            return out, y, lvl_loss, global_loss


    def log_confusion_matrix(self, pred, target, split_name, lvl):
        conf_mat = confusion_matrix(pred.flatten(), target.flatten(), labels=[i for i in range(self.out_classes)])
        fig = plt.figure(figsize=(20, 14))
        canvas = FigureCanvasAgg(fig)
        sns.heatmap(conf_mat, annot=True)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((int(height), int(width), 3))
        self.logger.experiment.add_image(f"confusion_matrix_{split_name}/lvl_{lvl}", image,
                                         dataformats="HWC", global_step=self.current_epoch)
        plt.close('all')

    def log_roc_auc(self, pred, target, split_name, lvl) -> None:
        pred = pred.flatten()
        target = target.flatten()
        score = roc_auc_score(y_true=target, y_score=pred)
        fpr, tpr, thresholds = roc_curve(target, pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        canvas = FigureCanvasAgg(display.figure_)
        canvas.draw()
        width, height = display.figure_.get_size_inches() * display.figure_.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((int(height), int(width), 3))
        self.logger.experiment.add_image(f"ROC_AUC_{split_name}/lvl_{lvl}", image, dataformats="HWC",
                                         global_step=self.current_epoch)
        self.log(f"roc_score_{split_name}/lvl_{lvl}", score)
        plt.close('all')

    def validation_epoch_end(self, validation_step_outputs):
        self._knn_classifier = None
        conc_val_outputs = []
        for curr_outputs in validation_step_outputs:
            out, y, lvl_loss, global_loss = zip(*curr_outputs)
            out_conc = []
            lvl_loss_conc = []
            for i in range(len(out[0])):
                out_conc.append(torch.concat([o[i] for o in out]).cpu().numpy())
                curr_lvl_loss = [l[i] for l in lvl_loss if l is not None]
                if curr_lvl_loss[0] is None:
                    curr_lvl_loss = None
                else:
                    curr_lvl_loss = [cll for cll in curr_lvl_loss if cll is not None]
                    curr_lvl_loss = float(torch.mean(torch.stack(curr_lvl_loss).flatten()).cpu().numpy())
                lvl_loss_conc.append(curr_lvl_loss)
            out = out_conc
            lvl_loss = lvl_loss_conc
            y = torch.concat(y).cpu().numpy()
            global_loss = [g for g in global_loss if isinstance(g, torch.Tensor)]
            global_loss = float(torch.mean(torch.stack(global_loss).flatten()).cpu().numpy())
            conc_val_outputs.append((out, y, lvl_loss, global_loss))

        for split_name, split_res in (("train", conc_val_outputs[0]), ("val", conc_val_outputs[1]), ("test", conc_val_outputs[2])):
            split_out, split_y, split_lvl_loss, split_global_loss = split_res
            self.log(f"{split_name}_loss/loss", split_global_loss)
            preds = [None] * len(split_lvl_loss)
            if self.phase == EModelPhase.TRAIN_EMBEDDING:
                if split_name == "train":
                    gpu_idx = int(self.cfg.enviroment.device.split("cuda:")[1])
                    self._knn_classifier = [KNNFaiss(k=5, gpu_idx=gpu_idx) for i in range(len(split_out))]
                idx_to_take = None
                if split_out[0].shape[0] > self.cfg.training.max_samples_for_evaluation:
                    idx_to_take = torch.randperm(split_out[0].shape[0])[:self.cfg.training.max_samples_for_evaluation]
                    split_y = split_y[idx_to_take]
                for i, lvl_out in enumerate(split_out):
                    if idx_to_take is not None:
                        lvl_out = lvl_out[idx_to_take]

                    if split_name == "train":
                        self._knn_classifier[i].fitModel(train_features=lvl_out, train_labels=split_y)
                    preds[i] = self._knn_classifier[i].predict(test_features=lvl_out)
            else:  # Final classification, the output is logits
                if self.cfg.dataset.multi_label_ds:
                    preds[-1] = (split_out[-1] > 0).astype(np.int)
                else:
                    preds[-1] = np.argmax(split_out[-1], axis=-1)

            split_acc = [None] * len(preds)
            split_f1 = [None] * len(preds)
            for i, (lvl_loss, lvl_pred) in enumerate(zip(split_lvl_loss, preds)):

                if lvl_loss is not None and lvl_pred is not None:
                    lvl_acc = np.mean(np.equal(lvl_pred.flatten(), split_y.flatten())) * 100
                    split_acc[i] = lvl_acc
                    if self.phase == EModelPhase.TRAIN_CLASSIFIER:
                        lvl_name = "final"
                    else:
                        lvl_name = i
                    self.log(f"loss_{split_name}/lvl_{lvl_name}", lvl_loss)
                    self.log(f"accuracy_{split_name}/lvl_{lvl_name}", lvl_acc)

                    if self.out_classes == 2 or self.cfg.dataset.multi_label_ds:
                        # Compute F1 score
                        f1s = f1_score(y_true=split_y, y_pred=np.reshape(lvl_pred, split_y.shape), average="micro")
                        self.log(f"f1_score_{split_name}/lvl_{lvl_name}", f1s)
                        split_f1[i] = f1s

                    # self.log_confusion_matrix(pred=lvl_pred, target=split_y, split_name=split_name, lvl=i)

                    # if self.out_classes == 2:
                    #     self.log_roc_auc(pred=lvl_pred, target=split_y, split_name=split_name, lvl=i)

                    self.logger.experiment.add_histogram(f'class_prediction_histogram/lvl_{i}', lvl_pred,
                                                         self.current_epoch)
            new_best_val = [False] * len(split_acc)
            for i, (lvl_acc, lvl_loss, f1s) in enumerate(zip(split_acc, split_lvl_loss, split_f1)):
                if lvl_loss is None and lvl_acc is None and f1s is None:
                    continue
                if self.phase == EModelPhase.TRAIN_CLASSIFIER:
                    lvl_name = "final"
                else:
                    lvl_name = f"{i}"
                if split_name == "test" and new_best_val[i]:
                    # Assumption that test is running after val
                        if lvl_acc is not None:
                            self._update_best_metric(metric=f"accuracy_{split_name}/lvl_{lvl_name}",
                                                     value=lvl_acc,
                                                     type="max",
                                                     force=True)
                        if lvl_loss is not None:
                            self._update_best_metric(metric=f"loss_{split_name}/lvl_{lvl_name}",
                                                     value=lvl_loss,
                                                     type="min",
                                                     force=True)

                        if f1s is not None:
                            self._update_best_metric(metric=f"f1_score_{split_name}/lvl_{lvl_name}",
                                                     value=f1s,
                                                     type="max",
                                                     force=True)
                else:
                    if lvl_loss:
                        self._update_best_metric(metric=f"loss_{split_name}/lvl_{lvl_name}",
                                                 value=lvl_loss,
                                                 type="min")
                    if lvl_acc is not None:
                        updated_acc = self._update_best_metric(metric=f"accuracy_{split_name}/lvl_{lvl_name}",
                                                               value=lvl_acc,
                                                               type="max")
                        if split_name == "val" and updated_acc:
                            new_best_val[i] = True
                            self.log(f"best_epoch/lvl_{lvl_name}", self.current_epoch)

                    if f1s is not None:
                        self._update_best_metric(metric=f"f1_score_{split_name}/lvl_{lvl_name}",
                                                 value=f1s,
                                                 type="max")
