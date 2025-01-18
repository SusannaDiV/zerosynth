import enum
from collections.abc import Sequence
from typing import TypedDict
import multiprocessing as mp
from functools import partial
import os
import sys
from pathlib import Path

import torch
import numpy as np
from rdkit import Chem
from skimage.util import view_as_blocks

# Set specific OpenBabel paths
conda_prefix = '/mnt/home/luost_local/micromamba/envs/sf'
os.environ['BABEL_LIBDIR'] = f'{conda_prefix}/lib/openbabel/3.1.0'
os.environ['BABEL_DATADIR'] = f'{conda_prefix}/share/openbabel/3.1.0'

# Now import OpenBabel
import openbabel as ob
from openbabel import pybel

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from skimage.util import view_as_blocks

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.mol import Molecule
from chemprojector.chem.reaction import Reaction
from chemprojector.chem.stack import Stack
from chemprojector.utils.image import draw_text, make_grid
from chemprojector.chem.tfbio_data import get_atom_stamp, get_shape

# Add shape cache
_shape_cache = {}
_atom_stamp = None  # Global atom stamp cache

def _init_worker():
    """Initialize worker process with global atom stamp"""
    global _atom_stamp
    resolution = 0.5
    _atom_stamp = get_atom_stamp(grid_resolution=resolution, max_dist=4.0)

def _generate_shape_patches(smiles: str) -> torch.Tensor:
    """Worker function to generate shape patches for a single molecule"""
    global _atom_stamp
    try:
        if _atom_stamp is None:
            _init_worker()
            
        # Force CPU computations for consistency
        with torch.device('cpu'):
            # Convert SMILES to OpenBabel molecule
            obmol = pybel.readstring("smi", smiles)
            if obmol is None:
                print(f"Failed to parse SMILES: {smiles}")
                return torch.zeros((343, 27), dtype=torch.float32)
            
            # Quick 3D generation with minimal optimization
            try:
                obmol.make3D(forcefield="uff", steps=50)  # Reduced steps
                obmol.localopt(forcefield="uff", steps=25)  # Minimal optimization
            except:
                try:
                    obmol.make3D(steps=25)  # Last resort with minimal steps
                except:
                    raise ValueError(f"Failed to generate 3D conformer for: {smiles}")
            
            # Convert to RDKit molecule for shape calculation
            mol_block = obmol.write("mol")
            rdmol = Chem.MolFromMolBlock(mol_block)
            if rdmol is None:
                raise ValueError("Failed to convert OpenBabel molecule to RDKit molecule")
            
            # Create cavity
            cavity = Chem.Mol(rdmol)
            if cavity is None:
                raise ValueError("Failed to create cavity")
            
            resolution = 0.5
            box_size = 15
            
            curr_cavity_shape = get_shape(
                mol=cavity,
                atom_stamp=_atom_stamp,
                grid_resolution=resolution,
                max_dist=box_size
            )
            
            if curr_cavity_shape is None or curr_cavity_shape.size == 0:
                raise ValueError("Failed to compute cavity shape")
                
            # Center and extract shape
            grid_size = 21
            start_idx = curr_cavity_shape.shape[0]//2 - grid_size//2
            end_idx = start_idx + grid_size
            
            centered_shape = curr_cavity_shape[
                start_idx:end_idx,
                start_idx:end_idx,
                start_idx:end_idx
            ]
            
            # Create patches
            shape_patches = view_as_blocks(centered_shape, (3, 3, 3))
            shape_patches = shape_patches.reshape(-1, 27)
            
            # Convert to tensor
            result = torch.from_numpy(shape_patches).to(torch.float32)
            return result
            
    except Exception as e:
        print(f"Error processing {smiles}: {str(e)}")
        return torch.zeros((343, 27), dtype=torch.float32)

# Create a process pool for parallel shape generation
_process_pool = None

def init_shape_generation(num_workers: int = None):
    """Initialize the process pool for parallel shape generation"""
    global _process_pool
    if _process_pool is None:
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        _process_pool = mp.Pool(num_workers, initializer=_init_worker)

def generate_shapes_parallel(smiles_list: list[str]) -> list[torch.Tensor]:
    """Generate shapes for multiple molecules in parallel"""
    global _process_pool
    if _process_pool is None:
        init_shape_generation()
    
    # Filter out already cached molecules
    uncached_smiles = [s for s in smiles_list if s not in _shape_cache]
    
    if uncached_smiles:
        # Generate shapes in parallel
        results = _process_pool.map(_generate_shape_patches, uncached_smiles)
        
        # Update cache with new results
        for smiles, shape in zip(uncached_smiles, results):
            _shape_cache[smiles] = shape
    
    # Return all shapes (from cache or newly generated)
    return [_shape_cache[s] for s in smiles_list]

def cleanup_shape_generation():
    """Clean up the process pool"""
    global _process_pool
    if _process_pool is not None:
        _process_pool.close()
        _process_pool.join()
        _process_pool = None

def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    mol_idx_seq: Sequence[int | None],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    fpindex: FingerprintIndex,
    encoder_type: str = "shape",
    device: torch.device = None,
    worker_id: int = None,
):
    # Convert product to SMILES
    product_smiles = Chem.MolToSmiles(product._rdmol, canonical=True)
    
    # Initialize worker if needed
    if worker_id is not None and _atom_stamp is None:
        _init_worker()
    
    # Check cache first
    if product_smiles in _shape_cache:
        shape_patches = _shape_cache[product_smiles]
    else:
        # Generate shape if not in cache
        shape_patches = _generate_shape_patches(product_smiles)
        _shape_cache[product_smiles] = shape_patches

    stack_feats = featurize_stack_actions(
        mol_idx_seq=mol_idx_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
        fpindex=fpindex,
    )

    data: ProjectionData = {
        "mol_seq": mol_seq,
        "rxn_seq": rxn_seq,
        "shape_patches": shape_patches,
        "token_types": stack_feats["token_types"],
        "rxn_indices": stack_feats["rxn_indices"],
        "reactant_fps": stack_feats["reactant_fps"],
        "token_padding_mask": stack_feats["token_padding_mask"],
    }
    return data


class TokenType(enum.IntEnum):
    END = 0
    START = 1
    REACTION = 2
    REACTANT = 3


class ProjectionData(TypedDict, total=False):
    # Encoder
    '''
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    '''
    # Shape encoder
    shape_patches: torch.Tensor
    # ph4_patches: torch.Tensor
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Molecule]
    rxn_seq: Sequence[Reaction | None]


class ProjectionBatch(TypedDict, total=False):
    # Shape encoder
    shape_patches: torch.Tensor  # [batch_size, 343, 27]
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Sequence[Molecule]]
    rxn_seq: Sequence[Sequence[Reaction | None]]
    # Shape encoder fields
    #shape: torch.Tensor  # [batch_size, box_size, box_size, box_size]
    # shape_patches: torch.Tensor  # [batch_size, num_patches, patch_size**3]
    #shape_padding_mask: torch.Tensor  # [batch_size, num_patches]
    # ph4_patches: torch.Tensor  # [batch_size, num_patches, patch_size**3]


def featurize_stack_actions(
    mol_idx_seq: Sequence[int | None],
    rxn_idx_seq: Sequence[int | None],
    end_token: bool,
    fpindex: FingerprintIndex,
) -> dict[str, torch.Tensor]:
    seq_len = len(mol_idx_seq) + 1  # Plus START token
    if end_token:
        seq_len += 1
    fp_dim = fpindex.fp_option.dim
    feats = {
        "token_types": torch.zeros([seq_len], dtype=torch.long),
        "rxn_indices": torch.zeros([seq_len], dtype=torch.long),
        "reactant_fps": torch.zeros([seq_len, fp_dim], dtype=torch.float),
        "token_padding_mask": torch.zeros([seq_len], dtype=torch.bool),
    }
    
    feats["token_types"][0] = TokenType.START
    
    for i, (mol_idx, rxn_idx) in enumerate(zip(mol_idx_seq, rxn_idx_seq), start=1):
        if rxn_idx is not None:
            feats["token_types"][i] = TokenType.REACTION
            feats["rxn_indices"][i] = rxn_idx
        elif mol_idx is not None:
            feats["token_types"][i] = TokenType.REACTANT
            _, mol_fp = fpindex[mol_idx]
            if mol_fp.dtype == np.uint8:
                mol_fp = mol_fp.astype(np.float32)
            feats["reactant_fps"][i] = torch.from_numpy(mol_fp)
            
    return feats


def featurize_stack(
    stack: Stack, 
    end_token: bool, 
    fpindex: FingerprintIndex, 
    encoder_type: str = "shape"
) -> dict[str, torch.Tensor]:
    return featurize_stack_actions(
        mol_idx_seq=stack.get_mol_idx_seq(),
        rxn_idx_seq=stack.get_rxn_idx_seq(),
        end_token=end_token,
        fpindex=fpindex,
    )


def draw_data(data: ProjectionData):
    im_list = [draw_text("START")]
    for m, r in zip(data["mol_seq"], data["rxn_seq"]):
        if r is not None:
            im_list.append(r.draw())
        else:
            im_list.append(m.draw())
    im_list.append(draw_text("END"))
    return make_grid(im_list)


def draw_batch(batch: ProjectionBatch):
    bsz = len(batch["mol_seq"])
    for b in range(bsz):
        im_list = [draw_text("START")]
        for m, r in zip(batch["mol_seq"][b], batch["rxn_seq"][b]):
            if r is not None:
                im_list.append(r.draw())
            else:
                im_list.append(m.draw())
        im_list.append(draw_text("END"))
        yield make_grid(im_list)
