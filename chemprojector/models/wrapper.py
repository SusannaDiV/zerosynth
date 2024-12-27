import pickle
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
import torch.nn as nn

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.matrix import ReactantReactionMatrix
from chemprojector.data.common import ProjectionBatch, draw_batch
from chemprojector.utils.train import get_optimizer, get_scheduler, sum_weighted_losses
from .encoder import get_encoder
from .chemprojector import ChemProjector, draw_generation_results


class ChemProjectorWrapper(pl.LightningModule):
    def __init__(self, config, args: dict | None = None):
        super().__init__()
        if config.version != 2:
            raise ValueError("Only version 2 is supported")
        self.save_hyperparameters(
            {
                "config": OmegaConf.to_container(config),
                "args": args or {},
            }
        )
        self.model = ChemProjector(config.model)
        self.is_shape_model = config.model.encoder_type == "shape"

    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])

    @property
    def args(self):
        return OmegaConf.create(self.hparams.get("args", {}))

    def setup(self, stage: str) -> None:
        super().setup(stage)
        
        # Only load chem data for non-shape models
        if not self.is_shape_model:
            with open(self.config.chem.rxn_matrix, "rb") as f:
                self.rxn_matrix: ReactantReactionMatrix = pickle.load(f)

            with open(self.config.chem.fpindex, "rb") as f:
                self.fpindex: FingerprintIndex = pickle.load(f)

    def configure_optimizers(self):
        if self.is_shape_model:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.train.optimizer.lr,
                weight_decay=self.config.train.optimizer.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.train.scheduler.factor,
                patience=self.config.train.scheduler.patience,
                min_lr=self.config.train.scheduler.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss"
            }
        else:
            optimizer = get_optimizer(self.config.train.optimizer, self.model)
            if "scheduler" in self.config.train:
                scheduler = get_scheduler(self.config.train.scheduler, optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val/loss",
                }
            return optimizer

    def training_step(self, batch, batch_idx: int):
        loss_dict, aux_dict = self.model.get_loss_shortcut(batch)
        if self.is_shape_model:
            loss = loss_dict["shape"]
            self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
            return loss
        else:
            loss_sum = sum_weighted_losses(loss_dict, self.config.train.loss_weights)
            self.log("train/loss", loss_sum, on_step=True, prog_bar=True, logger=True)
            self.log_dict({f"train/loss_{k}": v for k, v in loss_dict.items()}, on_step=True, logger=True)
            return loss_sum

    def validation_step(self, batch, batch_idx: int) -> Any:
        loss_dict, _ = self.model.get_loss_shortcut(batch)
        if self.is_shape_model:
            loss = loss_dict["shape"]
            self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
            return loss
        else:
            loss_weight = self.config.train.get("val_loss_weights", self.config.train.loss_weights)
            loss_sum = sum_weighted_losses(loss_dict, loss_weight)
            self.log("val/loss", loss_sum, on_step=False, prog_bar=True, logger=True, sync_dist=True)
            self.log_dict({f"val/loss_{k}": v for k, v in loss_dict.items()}, on_step=False, logger=True, sync_dist=True)
            return loss_sum
