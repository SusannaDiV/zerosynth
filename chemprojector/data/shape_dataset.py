import pickle as pkl
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import sys

# Add the ShapePretrainingDataset to the modules
from chemprojector.data.shape_pretraining_dataset import ShapePretrainingDataset
from chemprojector.data.collate import ShapeCollater

sys.modules['shape_pretraining_dataset'] = sys.modules['chemprojector.data.shape_pretraining_dataset']

class ShapeDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.dataset = pkl.load(f)
        self.data = self.dataset.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert numpy array to torch tensor if needed
        if not isinstance(sample['mol'], torch.Tensor):
            sample['mol'] = torch.from_numpy(sample['mol']).float()
        return {"shape": sample['mol']}

class ShapeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int = 4,
        patch_size: int = 3,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

    def setup(self, stage: str | None = None):
        self.dataset = ShapeDataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ShapeCollater(patch_size=self.patch_size),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ShapeCollater(patch_size=self.patch_size),
        ) 