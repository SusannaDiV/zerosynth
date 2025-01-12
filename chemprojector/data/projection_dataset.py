import os
import pickle
import random
from typing import cast, Dict, Optional
import numpy as np
import multiprocessing as mp

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset
from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.matrix import ReactantReactionMatrix
from chemprojector.chem.stack import create_stack_step_by_step
from chemprojector.utils.train import worker_init_fn
from skimage.util import view_as_blocks

# Set multiprocessing start method to spawn
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

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
    def __init__(self, max_num_atoms: int = 96, max_smiles_len: int = 192, max_num_tokens: int = 24, encoder_type: str = "shape"):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_smiles_len = max_smiles_len
        self.max_num_tokens = max_num_tokens
        self.encoder_type = encoder_type

        self.spec_shapes = {
            "shape_patches": collate_shape_patches#,
            #"ph4_patches": collate_shape_patches
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
        shapes_dir: str = "/home/luost_local/sdivita/zerosynth/data/processed/all/merged",
        molecules_per_batch: int = 120000,  # Increased for larger batches
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
        
        # Shape patches handling
        self._shapes_dir = shapes_dir
        self._molecules_per_batch = molecules_per_batch
        self._batch_files = [
            "temp_merged_round3_0.pkl",
            "temp_merged_round3_1.pkl"
        ]
        
        # Print debug info about batches
        print(f"\nUsing 2 merged batch files:")
        for i, f in enumerate(self._batch_files):
            print(f"  Batch {i}: {f}")
        print(f"Molecules per batch: {molecules_per_batch}")
        
        # Initialize cache
        self._batch_cache = {}
        self._max_cache_size = 2  # We only need to cache both files

    def _get_batch_id(self, mol_idx: int) -> tuple[Optional[int], Optional[int]]:
        """Calculate which batch file should contain the molecule and the local index within that batch
        
        Args:
            mol_idx: Global molecule index from the reaction matrix
            
        Returns:
            Tuple of (batch_id, local_idx) or (None, None) if invalid
            batch_id is which file to load (0 or 1)
            local_idx is the index within that file (0-119999)
        """
        batch_id = mol_idx // self._molecules_per_batch
        if batch_id < 0 or batch_id >= len(self._batch_files):
            print(f"Warning: mol_idx {mol_idx} maps to invalid batch {batch_id} (total batches: {len(self._batch_files)})")
            return None, None
            
        # Convert global index to local index within the batch
        local_idx = mol_idx % self._molecules_per_batch
        return batch_id, local_idx

    def _get_shape_patches(self, mol_idx: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get shape and ph4 patches for a molecule, loading the appropriate batch if needed"""
        batch_id, local_idx = self._get_batch_id(mol_idx)
        if batch_id is None:
            return None, None
            
        # Load batch if not in cache
        if batch_id not in self._batch_cache:
            batch_file = self._batch_files[batch_id]
            try:
                # Force CPU loading
                with open(os.path.join(self._shapes_dir, batch_file), 'rb') as f:
                    # Map CUDA tensors to CPU during load
                    batch_data = pickle.load(f, map_location='cpu')
                    print(f"\nLoaded merged batch {batch_id} ({batch_file}):")
                    print(f"  Contains {len(batch_data)} molecules")
                    print(f"  Index range: {min(batch_data.keys())} - {max(batch_data.keys())}")
                    # Debug: Look at structure of first molecule's patches
                    first_idx = next(iter(batch_data.keys()))
                    first_mol = batch_data[first_idx]
                    print(f"\nExample molecule structure (index {first_idx}):")
                    print(f"  Number of rotations: {len(first_mol)}")
                    print(f"  Shape patches shape: {first_mol[0][0].shape}")
                    print(f"  Ph4 patches shape: {first_mol[0][1].shape}")
                    self._batch_cache[batch_id] = batch_data
            except Exception as e:
                print(f"Error loading batch {batch_id}: {str(e)}")
                return None, None
        
        batch_data = self._batch_cache[batch_id]
        
        # Check if local_idx exists in the batch
        if local_idx not in batch_data:
            print(f"Warning: Molecule {mol_idx} has no shape patches:")
            print(f"  Global index: {mol_idx} = batch {batch_id} * {self._molecules_per_batch} + {local_idx}")
            print(f"  Batch contains {len(batch_data)} molecules")
            print(f"  Batch index range: {min(batch_data.keys())} - {max(batch_data.keys())}")
            return None, None
            
        # Get the patches for this molecule
        mol_patches = batch_data[local_idx]
        
        # Get random rotation index
        rotations = len(mol_patches)
        rotation_idx = random.randrange(rotations)
        
        # Debug: Print info about rotations
        if mol_idx % 1000 == 0:  # Only print occasionally
            print(f"\nMolecule {mol_idx} rotations:")
            print(f"  Total rotations available: {rotations}")
            print(f"  Selected rotation: {rotation_idx}")
            print(f"  Shape patches shape: {mol_patches[rotation_idx][0].shape}")
            print(f"  Ph4 patches shape: {mol_patches[rotation_idx][1].shape}")
        
        try:
            # Load tensors on CPU - each rotation has shape and ph4 patches
            # Ensure tensors are on CPU and detached from any computation graph
            shape_tensor = mol_patches[rotation_idx][0]
            ph4_tensor = mol_patches[rotation_idx][1]
            
            if isinstance(shape_tensor, torch.Tensor):
                shape_patches = shape_tensor.cpu().detach().float()
            else:
                shape_patches = torch.tensor(shape_tensor, dtype=torch.float, device='cpu')
                
            if isinstance(ph4_tensor, torch.Tensor):
                ph4_patches = ph4_tensor.cpu().detach().float()
            else:
                ph4_patches = torch.tensor(ph4_tensor, dtype=torch.float, device='cpu')
                
            return shape_patches, ph4_patches
            
        except Exception as e:
            print(f"Error processing patches for molecule {mol_idx}: {str(e)}")
            return None, None

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
                
                # Get product molecule index - find the last non-None index
                mol_idx = None
                for idx in reversed(mol_idx_seq_full):
                    if idx is not None:
                        mol_idx = idx
                        break
                
                if mol_idx is None:
                    print("Warning: No valid molecule index found in sequence")
                    data['shape_patches'] = torch.zeros((343, 27))
                    data['ph4_patches'] = torch.zeros((343, 162))
                    yield data
                    continue
                
                # Get shape patches from batch files
                shape_patches, ph4_patches = self._get_shape_patches(mol_idx)
                
                if shape_patches is not None and ph4_patches is not None:
                    data['shape_patches'] = shape_patches
                    data['ph4_patches'] = ph4_patches
                else:
                    # Default size for shape patches (343 patches, 27 features per patch)
                    print(f"Warning: Molecule {mol_idx} has no shape patches:")
                    print(f"  SMILES: {product.smiles}")
                    batch_id, local_idx = self._get_batch_id(mol_idx)
                    print(f"  Batch {batch_id}, Local index {local_idx}")
                    data['shape_patches'] = torch.zeros((343, 27))
                    data['ph4_patches'] = torch.zeros((343, 162))
                    
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
