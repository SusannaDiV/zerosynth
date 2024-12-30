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
from skimage.util import view_as_blocks

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

def get_shape_patches(shape, patch_size):
    assert shape.shape[0] % patch_size == 0 
    shape_patches = view_as_blocks(shape, (patch_size, patch_size, patch_size))
    return shape_patches

class Collater:
    def __init__(self, max_num_atoms: int = 96, max_smiles_len: int = 192, max_num_tokens: int = 24, encoder_type: str = "shape"):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_smiles_len = max_smiles_len
        self.max_num_tokens = max_num_tokens
        self.encoder_type = encoder_type

        self.spec_shapes = {
            "shape": collate_3d_grid,
            "shape_patches": collate_shape_patches
        }
        '''
        
        self.spec_atoms = {
            "atoms": collate_tokens,
            "bonds": collate_2d_tokens,
            "atom_padding_mask": collate_padding_masks,
        }
        self.spec_smiles = {"smiles": collate_tokens}
        '''
        self.spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }

    def __call__(self, data_list: list[ProjectionData]) -> ProjectionBatch:
        data_list_t = cast(list[dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self.spec_tokens, data_list_t, max_size=self.max_num_tokens),
            **apply_collate(self.spec_shapes, data_list_t, max_size=None),
            "mol_seq": [d["mol_seq"] for d in data_list],
            "rxn_seq": [d["rxn_seq"] for d in data_list],
        }
        '''
        # Only include structural data if not using shape encoder
        if self.encoder_type != "shape":
            batch.update({
                **apply_collate(self.spec_atoms, data_list_t, max_size=self.max_num_atoms),
                **apply_collate(self.spec_smiles, data_list_t, max_size=self.max_smiles_len),
            })
        '''
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
        encoder_type: str = "shape",
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_smiles_len = max_smiles_len
        self._max_num_reactions = max_num_reactions
        self._fpindex = fpindex
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length
        self.encoder_type = encoder_type

    def __len__(self) -> int:
        return self._virtual_length

    

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
                    encoder_type=self.encoder_type
                )
                #data["smiles"] = data["smiles"][: self._max_smiles_len]
                
                # Get shapes directly from fpindex
                mol_idx = mol_idx_seq_full[-1]  # Get index of product molecule
                if mol_idx in self._fpindex.shapes:
                    # Get random rotation index
                    rotation_idx = random.randrange(len(self._fpindex.shapes[mol_idx]))
                    # Use same rotation index for both shape and patches
                    shape = self._fpindex.shapes[mol_idx][rotation_idx]
                    shape_patches = self._fpindex.shape_patches[mol_idx][rotation_idx]
                    
                    data['shape'] = torch.tensor(shape, dtype=torch.float)
                    data['shape_patches'] = torch.tensor(shape_patches, dtype=torch.float)
                else:
                    data['shape'] = torch.zeros((21, 21, 21))
                    data['shape_patches'] = torch.zeros((343, 27))
                    
                yield data


class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_options = kwargs
        self.encoder_type = config.model.encoder_type

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
            collate_fn=Collater(encoder_type=self.encoder_type),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(encoder_type=self.encoder_type),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
