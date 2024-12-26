import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from chemprojector.models.encoder import get_encoder

class ShapeWrapper(pl.LightningModule):
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
        self.encoder = get_encoder(config.model)

    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])

    def configure_optimizers(self):
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

    def training_step(self, batch, batch_idx):
        encoded, padding_mask = self.encoder(batch)
        # For now, just use a simple reconstruction loss
        loss = torch.nn.functional.mse_loss(encoded, batch["shapes"])
        
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        encoded, padding_mask = self.encoder(batch)
        loss = torch.nn.functional.mse_loss(encoded, batch["shapes"])
        
        self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return loss 