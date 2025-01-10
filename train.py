import os

import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers, strategies
from torch.profiler import profile, record_function, ProfilerActivity
import time

from chemprojector.data.projection_dataset import ProjectionDataModule
from chemprojector.models.wrapper import ChemProjectorWrapper
from chemprojector.utils.misc import (
    get_config_name,
    get_experiment_name,
    get_experiment_version,
)
from chemprojector.utils.vc import get_vc_info

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


class ProfilingCallback(callbacks.Callback):
    def __init__(self):
        self.batch_start_time = None
        self.data_load_times = []
        self.forward_times = []
        self.backward_times = []
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        if batch_idx == 0:  # Start profiler on first batch
            self.profiler = profile(activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ], with_stack=True, record_shapes=True)
            self.profiler.start()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 10:  # Profile first 10 batches
            self.profiler.step()
            if batch_idx == 9:  # Print profile after 10 batches
                print("\nPROFILER RESULTS\n")
                print(self.profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20))
                self.profiler.stop()
        
        # Log timing metrics
        batch_time = time.time() - self.batch_start_time
        trainer.logger.log_metrics({
            'batch_time': batch_time,
        }, step=trainer.global_step)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--seed", type=int, default=42)
@click.option("--debug", is_flag=True)
@click.option("--batch-size", "-b", type=int, default=256)
@click.option("--num-workers", type=int, default=4)
@click.option("--devices", type=int, default=4)
@click.option("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", 1)))
@click.option("--num-sanity-val-steps", type=int, default=1)
@click.option("--log-dir", type=click.Path(dir_okay=True, file_okay=False), default="./logs")
@click.option("--resume", type=click.Path(exists=True, dir_okay=False), default=None)
def main(
    config_path: str,
    seed: int,
    debug: bool,
    batch_size: int,
    num_workers: int,
    devices: int,
    num_nodes: int,
    num_sanity_val_steps: int,
    log_dir: str,
    resume: str | None,
):
    if batch_size % devices != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    batch_size_per_process = batch_size // devices

    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed)

    config = OmegaConf.load(config_path)
    config_name = get_config_name(config_path)
    vc_info = get_vc_info()
    vc_info.disallow_changes(debug)
    exp_name = get_experiment_name(config_name, vc_info.display_version, vc_info.committed_at)
    exp_ver = get_experiment_version()

    # Dataloaders
    datamodule = ProjectionDataModule(
        config,
        batch_size=batch_size_per_process,
        num_workers=num_workers,
        **config.data,
    )

    # Model
    model = ChemProjectorWrapper(config)

    # Add profiling callback
    profiling_callback = ProfilingCallback()
    
    # Train with profiling
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategies.DDPStrategy(static_graph=True),
        num_sanity_val_steps=num_sanity_val_steps,
        gradient_clip_val=config.train.max_grad_norm,
        log_every_n_steps=1,
        max_steps=config.train.max_iters,
        callbacks=[
            callbacks.ModelCheckpoint(save_last=True, monitor="val/loss", mode="min", save_top_k=5),
            callbacks.LearningRateMonitor(logging_interval="step"),
            profiling_callback,  # Add profiling callback
        ],
        logger=[
            loggers.TensorBoardLogger(log_dir, name=exp_name, version=exp_ver),
        ],
        val_check_interval=config.train.val_freq,
        limit_val_batches=4,
        enable_progress_bar=True,  # Enable to see real-time progress
    )

    # Add memory profiling before fit
    print("\nInitial CUDA memory usage:")
    print(torch.cuda.memory_summary())
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume)
    
    # Print final profiling info
    print("\nFinal CUDA memory usage:")
    print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
