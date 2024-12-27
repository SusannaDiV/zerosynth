import os
import pickle
import random
from typing import cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.matrix import ReactantReactionMatrix
from chemprojector.chem.stack import create_stack_step_by_step
from chemprojector.utils.train import worker_init_fn

from .collate import (
    apply_collate,
    collate_1d_features,
    collate_2d_tokens,
    collate_padding_masks,
    collate_tokens,
    collate_3d_grid,
    collate_shape_patches,
)
from .common import ProjectionBatch, ProjectionData, create_data

import sys
from chemprojector.chem.fpindex import FingerprintIndex
sys.modules['__main__'].FingerprintIndex = FingerprintIndex


class Collater:
    def __init__(self, max_num_atoms: int = 96, max_smiles_len: int = 192, max_num_tokens: int = 24):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_smiles_len = max_smiles_len
        self.max_num_tokens = max_num_tokens

        self.spec_shapes = {
            "shape": collate_3d_grid,
            "shape_patches": collate_shape_patches
        }
        
        self.spec_atoms = {
            "atoms": collate_tokens,
            "bonds": collate_2d_tokens,
            "atom_padding_mask": collate_padding_masks,
        }
        self.spec_smiles = {"smiles": collate_tokens}
        self.spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }

    def __call__(self, data_list: list[ProjectionData]) -> ProjectionBatch:
        data_list_t = cast(list[dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self.spec_atoms, data_list_t, max_size=self.max_num_atoms),
            **apply_collate(self.spec_smiles, data_list_t, max_size=self.max_smiles_len),
            **apply_collate(self.spec_tokens, data_list_t, max_size=self.max_num_tokens),
            **apply_collate(self.spec_shapes, data_list_t, max_size=None),
            "mol_seq": [d["mol_seq"] for d in data_list],
            "rxn_seq": [d["rxn_seq"] for d in data_list],
        }
        return cast(ProjectionBatch, batch)


class ProjectionDataset(IterableDataset[ProjectionData]):
    def __init__(
        self,
        reaction_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        virtual_length: int = 65536,
        max_num_atoms: int = 80,
        max_smiles_len: int = 192,
        max_num_reactions: int = 5,
        init_stack_weighted_ratio: float = 0.0,
        shape_data: list | None = None,
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_smiles_len = max_smiles_len
        self._max_num_reactions = max_num_reactions
        self._fpindex = fpindex
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length
        self.shape_data = shape_data

    def __len__(self) -> int:
        return self._virtual_length

    def _create_patches(self, shape_grid, patch_size=3):
        """Convert 3D grid into patches"""
        shape = torch.tensor(shape_grid, dtype=torch.float)
        patches = shape.unfold(0, patch_size, patch_size)\
                      .unfold(1, patch_size, patch_size)\
                      .unfold(2, patch_size, patch_size)
        patches = patches.reshape(-1, patch_size**3)
        return patches

    def __iter__(self):
        while True:
            for stack in create_stack_step_by_step(
                self._reaction_matrix,
                max_num_reactions=self._max_num_reactions,
                max_num_atoms=self._max_num_atoms,
                init_stack_weighted_ratio=self._init_stack_weighted_ratio,
            ):
                mol_seq_full = stack.mols
                mol_idx_seq_full = stack.get_mol_idx_seq()
                rxn_seq_full = stack.rxns
                rxn_idx_seq_full = stack.get_rxn_idx_seq()
                product = random.choice(list(stack.get_top()))
                data = create_data(
                    product=product,
                    mol_seq=mol_seq_full,
                    mol_idx_seq=mol_idx_seq_full,
                    rxn_seq=rxn_seq_full,
                    rxn_idx_seq=rxn_idx_seq_full,
                    fpindex=self._fpindex,
                )
                data["smiles"] = data["smiles"][: self._max_smiles_len]
                
                # Add shape data if available
                if self.shape_data is not None:
                    shape_item = random.choice(self.shape_data)
                    shape_tensor = torch.tensor(shape_item['shape'], dtype=torch.float)
                    data['shape'] = shape_tensor
                    data['shape_patches'] = self._create_patches(shape_item['shape'])
                else:
                    # Add empty tensors if no shape data
                    print("empty")
                    data['shape'] = torch.zeros((1, 1, 1))  # Minimal 3D tensor
                    data['shape_patches'] = torch.zeros((1, 27))  # For 3x3x3 patches
                    
                yield data


class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        shape_data_path: str = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation/dataset1.pkl",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shape_data_path = shape_data_path
        self.dataset_options = kwargs

    def setup(self, stage: str | None = None) -> None:
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("The trainer is missing.")

        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)

        # Optionally load shape data if path is provided
        if self.shape_data_path is not None:
            with open(self.shape_data_path, "rb") as f:
                shape_data = pickle.load(f)
            self.dataset_options['shape_data'] = shape_data

        self.train_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.config.train.val_freq * self.batch_size,
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.batch_size,
            **self.dataset_options,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
