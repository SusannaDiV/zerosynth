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
    collate_shape_patches
)
from .common import ProjectionBatch, ProjectionData, create_data

import sys
from chemprojector.chem.fpindex import FingerprintIndex
sys.modules['__main__'].FingerprintIndex = FingerprintIndex

class Collater:
    def __init__(self, encoder_type: str = "shape", max_num_tokens: int = 24):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.encoder_type = encoder_type

        self.spec_shape = {
            "shape_patches": collate_shape_patches,
        }
        self.spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }

    def __call__(self, data_list: list[ProjectionData]) -> ProjectionBatch:
        data_list_t = cast(list[dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self.spec_shape, data_list_t, max_size=None),
            **apply_collate(self.spec_tokens, data_list_t, max_size=self.max_num_tokens),
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
        encoder_type: str = "shape",
        device: torch.device = None,
    ) -> None:
        super().__init__()
        
        # Debug fpindex contents
        print("\nFPIndex Debug:")
        print(f"Total molecules in fpindex: {len(fpindex._shape_patches)}")
        if len(fpindex._shape_patches) > 0:
            first_idx = next(iter(fpindex._shape_patches))
            first_patches = fpindex._shape_patches[first_idx]
            print(f"First molecule patches shape: {first_patches.shape}")
            print(f"First molecule patches non-zeros: {torch.count_nonzero(first_patches).item()}")
            print(f"Sample of indices: {list(fpindex._shape_patches.keys())[:5]}")
        else:
            print("WARNING: No shape patches found in fpindex!")
        
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_smiles_len = max_smiles_len
        self._max_num_reactions = max_num_reactions
        self._fpindex = fpindex
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length
        self.encoder_type = encoder_type
        self.device = torch.device('cpu')        
        print("ProjectionDataset initialization complete")

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
                
                # Get a random product from the top of the stack
                product = random.choice(list(stack.get_top()))
                
                # Find the index of the product molecule by matching SMILES
                # First try exact object identity
                product_idx = None
                for i, mol in enumerate(mol_seq_full):
                    if mol is product:
                        product_idx = i
                        break
                
                # If not found by identity, try SMILES matching
                if product_idx is None:
                    for i, mol in enumerate(mol_seq_full):
                        if mol.smiles == product.smiles:
                            product_idx = i
                            break
                
                # If still not found, use last non-None molecule index
                if product_idx is None:
                    ##print(f"Warning: Product molecule {product.smiles} not found in sequence, using last molecule")
                    # Find the last non-None molecule index
                    for i in range(len(mol_idx_seq_full) - 1, -1, -1):
                        if mol_idx_seq_full[i] is not None:
                            product_idx = i
                            break
                
                mol_idx = mol_idx_seq_full[product_idx] if product_idx is not None else None
                
                # Create data for the product molecule
                data = create_data(
                    product=product,
                    mol_seq=mol_seq_full,
                    mol_idx_seq=mol_idx_seq_full,
                    rxn_seq=rxn_seq_full,
                    rxn_idx_seq=rxn_idx_seq_full,
                    fpindex=self._fpindex,
                    encoder_type=self.encoder_type
                )
                
                # Override shape patches with correct molecule index
                if mol_idx is not None and mol_idx in self._fpindex._shape_patches:
                    data['shape_patches'] = self._fpindex._shape_patches[mol_idx].cpu()  # Ensure CPU tensor
                    ##print(f"Shape patches found for molecule {mol_idx}")
                    ###print("nonZeros projection")

                else:
                    # Only print warning for actual molecules (not reaction tokens)
                    ##if mol_idx is not None:
                        ##print(f"No shape patches found for molecule {mol_idx}")
                    data['shape_patches'] = torch.zeros((343, 27), dtype=torch.float32).cpu()  # (7^3, 3^3)
                
                # Extra safety check for shape patches
                if 'shape_patches' in data and data['shape_patches'].device.type != 'cpu':
                    data['shape_patches'] = data['shape_patches'].cpu()
                            
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
        # Set default device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

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

        from tqdm import tqdm
        import mmap
        import struct
        
        print("Loading reaction matrix...")
        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        print("Loading fingerprint index...")
        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)
            # Convert all shape patches to float32
            for idx in fpindex._shape_patches:
                if isinstance(fpindex._shape_patches[idx], torch.Tensor):
                    fpindex._shape_patches[idx] = fpindex._shape_patches[idx].to(torch.float32).cpu()

        print("Creating train dataset...")
        with tqdm(total=100, desc="Initializing train dataset") as pbar:
            self.train_dataset = ProjectionDataset(
                reaction_matrix=rxn_matrix,
                fpindex=fpindex,
                virtual_length=self.config.train.val_freq * self.batch_size,
                **self.dataset_options,
            )
            pbar.update(100)
        
        print("Creating validation dataset...")
        with tqdm(total=100, desc="Initializing val dataset") as pbar:
            self.val_dataset = ProjectionDataset(
                reaction_matrix=rxn_matrix,
                fpindex=fpindex,
                virtual_length=self.batch_size,
                **self.dataset_options,
            )
            pbar.update(100)
        
        print("Setup complete!")
        
    def train_dataloader(self):
        print("Initializing train dataloader...")
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(encoder_type=self.encoder_type),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
        print("Train dataloader ready")
        return loader

    def val_dataloader(self):
        print("Initializing validation dataloader...")
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(encoder_type=self.encoder_type),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
        print("Validation dataloader ready")
        return loader