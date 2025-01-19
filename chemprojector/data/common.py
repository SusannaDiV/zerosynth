import enum
from collections.abc import Sequence
from typing import TypedDict
import multiprocessing as mp
from functools import partial
import os
import sys
from pathlib import Path
from contextlib import contextmanager
import warnings
import io
import tempfile
import subprocess
import pathlib
import hashlib
from datetime import datetime

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
from chemprojector.chem.tfbio_data import (
    get_atom_stamp, 
    get_shape, 
    make_grid_mol,
    ATOMIC_NUMBER,
    ATOMIC_NUMBER_REVERSE
)

# Add shape cache
_shape_cache = {}
_atom_stamp = None  # Global atom stamp cache

# Pharmacophore SMARTS patterns from Pharmer software
ARO_SMARTS = ["a1aaaaa1", "a1aaaa1"]

HBD_SMARTS = ["[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]",
              "[#8!H0&!$([OH][C,S,P]=O)]",
              "[#16!H0]"]

HBA_SMARTS = ["[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]",
              "[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]"]

POS_SMARTS = ["[+,+2,+3,+4]",
              "[$(CC)](=N)N",  # amidine
              "[$(C(N)(N)=N)]",  # guanidine
              "[$(n1cc[nH]c1)]"]

NEG_SMARTS = ["[-,-2,-3,-4]",
              "C(=O)[O-,OH,OX1]",
              "[$([S,P](=O)[O-,OH,OX1])]",
              "c1[nH1]nnn1",
              "c1nn[nH1]n1",
              "C(=O)N[OH1,O-,OX1]",
              "C(=O)N[OH1,O-]",
              "CO(=N[OH1,O-])",
              "[$(N-[SX4](=O)(=O)[CX4](F)(F)F)]"]

HYD_SMARTS = ["a1aaaaa1",
              "a1aaaa1",
              "[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
              "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
              "*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
              "[C&r3]1~[C&r3]~[C&r3]1",
              "[C&r4]1~[C&r4]~[C&r4]~[C&r4]1",
              "[C&r5]1~[C&r5]~[C&r5]~[C&r5]~[C&r5]1",
              "[C&r6]1~[C&r6]~[C&r6]~[C&r6]~[C&r6]~[C&r6]1",
              "[C&r7]1~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]1",
              "[C&r8]1~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]1",
              "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
              "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
              "[$([CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
              "[$([S]~[#6])&!$(S~[!#6])]"]

def find_matches(obmol, smarts_pattern):
    """Find all matches for a SMARTS pattern in an OpenBabel molecule"""
    matches = []
    pattern = ob.OBSmartsPattern()
    pattern.Init(smarts_pattern)
    pattern.Match(obmol.OBMol)
    
    if pattern.NumMatches() > 0:
        match_list = pattern.GetMapList()
        for match in match_list:
            # Calculate center of matched atoms
            x_sum = y_sum = z_sum = 0
            for atom_idx in match:
                atom = obmol.OBMol.GetAtom(atom_idx)
                x_sum += atom.GetX()
                y_sum += atom.GetY()
                z_sum += atom.GetZ()
            n_atoms = len(match)
            center = (x_sum/n_atoms, y_sum/n_atoms, z_sum/n_atoms)
            matches.append(center)
    return matches

def find_features(obmol, smarts_list):
    """Find all pharmacophore features for a list of SMARTS patterns"""
    all_matches = []
    for smarts in smarts_list:
        matches = find_matches(obmol, smarts)
        all_matches.extend(matches)
    return all_matches

def cluster_hydrophobic(points, cutoff=2.0):
    """Cluster hydrophobic points that are within cutoff distance"""
    if not points:
        return []
    
    n = len(points)
    clusters = list(range(n))
    
    # Cluster points
    for i in range(n):
        p1 = points[i]
        cluster_id = clusters[i]
        for j in range(i+1, n):
            p2 = points[j]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dz = p1[2] - p2[2]
            if (dx*dx + dy*dy + dz*dz) <= cutoff*cutoff:
                clusters[j] = cluster_id
    
    # Average points in each cluster
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(points[i])
    
    # Calculate cluster centers
    centers = []
    for points in cluster_dict.values():
        x_sum = y_sum = z_sum = 0
        for x, y, z in points:
            x_sum += x
            y_sum += y
            z_sum += z
        n = len(points)
        centers.append((x_sum/n, y_sum/n, z_sum/n))
    
    return centers

def get_pharmacophore_features(obmol):
    """Get all pharmacophore features for a molecule"""
    coords = []
    types = []
    
    # Find all feature types
    aro = find_features(obmol, ARO_SMARTS)
    hbd = find_features(obmol, HBD_SMARTS)
    hba = find_features(obmol, HBA_SMARTS)
    pos = find_features(obmol, POS_SMARTS)
    neg = find_features(obmol, NEG_SMARTS)
    hyd = find_features(obmol, HYD_SMARTS)
    hyd = cluster_hydrophobic(hyd)  # Cluster hydrophobic features
    
    # Add features with their types
    for feat in aro:
        coords.append(feat)
        types.append(0)  # Aromatic
    for feat in hbd:
        coords.append(feat)
        types.append(1)  # Donor
    for feat in hba:
        coords.append(feat)
        types.append(2)  # Acceptor
    for feat in pos:
        coords.append(feat)
        types.append(3)  # Positive
    for feat in neg:
        coords.append(feat)
        types.append(4)  # Negative
    for feat in hyd:
        coords.append(feat)
        types.append(5)  # Hydrophobic
    
    return torch.tensor(coords, dtype=torch.float32), torch.tensor(types, dtype=torch.long)

def _init_worker():
    """Initialize worker process with global atom stamp"""
    global _atom_stamp
    resolution = 0.5
    _atom_stamp = get_atom_stamp(grid_resolution=resolution, max_dist=4.0)

def get_shape_from_obmol(obmol, atom_stamp, grid_resolution, max_dist):
    """Generate shape directly from OpenBabel molecule"""
    # Get coordinates and atom types directly from OpenBabel
    coords = np.array([atom.coords for atom in obmol.atoms])
    atom_types = [atom.atomicnum for atom in obmol.atoms]
    
    # Create features array (one-hot encoding of atom types)
    features = np.array(atom_types)[:, None]  # Convert to column vector
    
    # Use make_grid from tfbio_data
    grid, atomic2grid = make_grid_mol(coords, features, grid_resolution, max_dist)
    shape = np.zeros(grid[0, :, :, :, 0].shape)
    
    for tup in atomic2grid:
        atomic_number = int(tup[0])
        if atomic_number not in ATOMIC_NUMBER_REVERSE:
            continue  # Skip atoms not in our mapping
        stamp = atom_stamp[ATOMIC_NUMBER_REVERSE[atomic_number]]
        for grid_ijk in atomic2grid[tup]:
            i, j, k = grid_ijk
            
            x_left = max(0, i - stamp.shape[0] // 2)
            x_right = min(shape.shape[0] - 1, i + stamp.shape[0] // 2)
            x_l = i - x_left
            x_r = x_right - i
            
            y_left = max(0, j - stamp.shape[1] // 2)
            y_right = min(shape.shape[1] - 1, j + stamp.shape[1] // 2)
            y_l = j - y_left
            y_r = y_right - j
            
            z_left = max(0, k - stamp.shape[2] // 2)
            z_right = min(shape.shape[2] - 1, k + stamp.shape[2] // 2)
            z_l = k - z_left
            z_r = z_right - k
            
            mid = stamp.shape[0] // 2
            shape_part = shape[x_left:x_right + 1, y_left:y_right + 1, z_left:z_right + 1]
            stamp_part = stamp[mid - x_l:mid + x_r + 1, mid - y_l:mid + y_r + 1, mid - z_l:mid + z_r + 1]
            
            shape_part += stamp_part
    
    shape[shape > 0] = 1
    return shape

def _generate_shape_patches(smiles: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Worker function to generate shape and pharmacophore patches for a single molecule"""
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
                return (torch.zeros((343, 27), dtype=torch.float32), 
                       torch.zeros((343, 27 * 6), dtype=torch.float32),
                       torch.zeros(840, dtype=torch.float32))
            
            # Generate 3D conformer
            try:
                obmol.make3D(forcefield="uff", steps=50)
                obmol.localopt(forcefield="uff", steps=25)
            except:
                try:
                    obmol.make3D(steps=25)
                except:
                    raise ValueError(f"Failed to generate 3D conformer for: {smiles}")
            
            # Get pharmacophore features using SMARTS patterns
            ph4_coords, ph4_types = get_pharmacophore_features(obmol)
            
            # Generate ACP4 fingerprint here
            acp4_fp = generate_acp4_fingerprint(obmol)
            
            # Create shape
            resolution = 0.5
            box_size = 15
            grid_size = int(2 * box_size / resolution) + 1
            
            # Create pharmacophore grid
            ph4_grid = torch.zeros((grid_size, grid_size, grid_size, 6), device='cpu')
            
            # Transform to grid space with more precise binning
            if len(ph4_coords) > 0:
                grid_coords = (ph4_coords + box_size) / resolution
                grid_coords = torch.round(grid_coords * 1000) / 1000  # Round to 3 decimal places
                bin_indices = grid_coords.floor().long()  # Consistent floor operation
                
                # Bin pharmacophore features
                for feat_idx in range(len(ph4_coords)):
                    x, y, z = bin_indices[feat_idx]
                    feat_type = ph4_types[feat_idx]
                    if (0 <= x < grid_size) and (0 <= y < grid_size) and (0 <= z < grid_size):
                        ph4_grid[x, y, z, feat_type] += 1
            
            # Create shape
            curr_cavity_shape = get_shape_from_obmol(
                obmol=obmol,
                atom_stamp=_atom_stamp,
                grid_resolution=resolution,
                max_dist=box_size
            )
            
            # Center and extract both grids
            grid_size = 21
            start_idx = curr_cavity_shape.shape[0]//2 - grid_size//2
            end_idx = start_idx + grid_size
            
            centered_shape = curr_cavity_shape[
                start_idx:end_idx,
                start_idx:end_idx,
                start_idx:end_idx
            ]
            
            centered_ph4 = ph4_grid[
                start_idx:end_idx,
                start_idx:end_idx,
                start_idx:end_idx
            ]
            
            # Convert to numpy for view_as_blocks
            centered_shape_np = centered_shape.cpu().numpy() if isinstance(centered_shape, torch.Tensor) else centered_shape
            centered_ph4_np = centered_ph4.cpu().numpy() if isinstance(centered_ph4, torch.Tensor) else centered_ph4
            
            # Create patches
            shape_patches = view_as_blocks(centered_shape_np, (3, 3, 3))
            shape_patches = shape_patches.reshape(-1, 27)
            
            ph4_patches = view_as_blocks(centered_ph4_np, (3, 3, 3, 6))
            ph4_patches = ph4_patches.reshape(-1, 27 * 6)
            
            # Convert back to tensors
            shape_patches = torch.from_numpy(shape_patches).float()
            ph4_patches = torch.from_numpy(ph4_patches).float()
            
            return shape_patches, ph4_patches, acp4_fp
            
    except Exception as e:
        print(f"Error processing {smiles}: {str(e)}")
        return (torch.zeros((343, 27), dtype=torch.float32), 
                torch.zeros((343, 27 * 6), dtype=torch.float32),
                torch.zeros(840, dtype=torch.float32))

# Create a process pool for parallel shape generation
_process_pool = None

def init_shape_generation(num_workers: int = None):
    """Initialize the process pool for parallel shape generation"""
    global _process_pool
    if _process_pool is None:
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        _process_pool = mp.Pool(num_workers, initializer=_init_worker)

def generate_shapes_parallel(smiles_list: list[str]) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate shapes, ph4 patches, and ACP4 fingerprints for multiple molecules in parallel"""
    global _process_pool
    if _process_pool is None:
        init_shape_generation()
    
    # Filter out already cached molecules
    uncached_smiles = [s for s in smiles_list if s not in _shape_cache]
    
    if uncached_smiles:
        # Generate features in parallel
        results = _process_pool.map(_generate_shape_patches, uncached_smiles)
        
        # Update cache with new results
        for smiles, result in zip(uncached_smiles, results):
            _shape_cache[smiles] = result  # Store tuple of (shape_patches, ph4_patches, acp4_fp)
    
    # Return all features (from cache or newly generated)
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
        shape_patches, ph4_patches, acp4_fp = _shape_cache[product_smiles]
    else:
        # Generate features if not in cache
        shape_patches, ph4_patches, acp4_fp = _generate_shape_patches(product_smiles)
        _shape_cache[product_smiles] = (shape_patches, ph4_patches, acp4_fp)

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
        "ph4_patches": ph4_patches,
        "acp4_fp": acp4_fp,  # Added ACP4 fingerprint
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
    ph4_patches: torch.Tensor
    acp4_fp: torch.Tensor
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
    ph4_patches: torch.Tensor
    acp4_fp: torch.Tensor
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

import matplotlib.pyplot as plt
from datetime import datetime

def visualize_shape_generation(smiles: str, save_path: str = "shapes"):
    """Visualize the shape generation process for a single molecule"""
    # Initialize if needed
    global _atom_stamp
    if _atom_stamp is None:
        _init_worker()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate the shape using the exact same code as _generate_shape_patches
    with torch.device('cpu'):
        # Convert SMILES to OpenBabel molecule
        obmol = pybel.readstring("smi", smiles)
        
        # Quick 3D generation with minimal optimization
        obmol.make3D(forcefield="uff", steps=50)
        obmol.localopt(forcefield="uff", steps=25)
        
        # Get pharmacophore features using SMARTS patterns
        ph4_coords, ph4_types = get_pharmacophore_features(obmol)
        
        # Create shape and ph4 grid
        resolution = 0.5
        box_size = 15
        grid_size = int(2 * box_size / resolution) + 1
        
        # Create pharmacophore grid
        ph4_grid = torch.zeros((grid_size, grid_size, grid_size, 6), device='cpu')
        
        # Transform to grid space with more precise binning
        if len(ph4_coords) > 0:
            grid_coords = (ph4_coords + box_size) / resolution
            grid_coords = torch.round(grid_coords * 1000) / 1000  # Round to 3 decimal places
            bin_indices = grid_coords.floor().long()  # Consistent floor operation
            
            # Bin pharmacophore features
            for feat_idx in range(len(ph4_coords)):
                x, y, z = bin_indices[feat_idx]
                feat_type = ph4_types[feat_idx]
                if (0 <= x < grid_size) and (0 <= y < grid_size) and (0 <= z < grid_size):
                    ph4_grid[x, y, z, feat_type] += 1
        
        # Create shape
        curr_cavity_shape = get_shape_from_obmol(
            obmol=obmol,
            atom_stamp=_atom_stamp,
            grid_resolution=resolution,
            max_dist=box_size
        )
        
        # Center and extract shape
        grid_size = 21
        start_idx = curr_cavity_shape.shape[0]//2 - grid_size//2
        end_idx = start_idx + grid_size
        
        centered_shape = curr_cavity_shape[
            start_idx:end_idx,
            start_idx:end_idx,
            start_idx:end_idx
        ]
        
        centered_ph4 = ph4_grid[
            start_idx:end_idx,
            start_idx:end_idx,
            start_idx:end_idx
        ]
        
        # Convert tensors to numpy for view_as_blocks
        centered_shape_np = centered_shape.cpu().numpy() if isinstance(centered_shape, torch.Tensor) else centered_shape
        centered_ph4_np = centered_ph4.cpu().numpy() if isinstance(centered_ph4, torch.Tensor) else centered_ph4
        
        # Create patches
        shape_patches = view_as_blocks(centered_shape_np, (3, 3, 3))
        shape_patches = shape_patches.reshape(-1, 27)
        
        ph4_patches = view_as_blocks(centered_ph4_np, (3, 3, 3, 6))
        ph4_patches = ph4_patches.reshape(-1, 27 * 6)
        
        # Convert back to tensors
        shape_patches = torch.from_numpy(shape_patches).float()
        ph4_patches = torch.from_numpy(ph4_patches).float()
        
        # Save visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot shape
        fig = plt.figure(figsize=(15, 5))
        
        # Plot molecular shape
        ax1 = fig.add_subplot(131, projection='3d')
        x, y, z = np.where(centered_shape > 0)
        ax1.scatter(x, y, z, c='blue', alpha=0.1, marker='s')
        ax1.set_title('Molecular Shape')
        
        # Plot pharmacophore features
        ax2 = fig.add_subplot(132, projection='3d')
        feature_names = ['Aromatic', 'H-Bond Donor', 'H-Bond Acceptor', 
                        'Positive', 'Negative', 'Hydrophobic']
        colors = ['purple', 'blue', 'red', 'green', 'orange', 'gray']
        
        for feat_type in range(6):
            feat_coords = np.where(centered_ph4[..., feat_type] > 0)
            if len(feat_coords[0]) > 0:
                ax2.scatter(feat_coords[0], feat_coords[1], feat_coords[2], 
                          c=colors[feat_type], label=feature_names[feat_type],
                          alpha=0.6, marker='o')
        ax2.legend()
        ax2.set_title('Pharmacophore Features')
        
        # Plot combined visualization
        ax3 = fig.add_subplot(133, projection='3d')
        # Plot shape first
        x, y, z = np.where(centered_shape > 0)
        ax3.scatter(x, y, z, c='blue', alpha=0.1, marker='s')
        # Plot features on top
        for feat_type in range(6):
            feat_coords = np.where(centered_ph4[..., feat_type] > 0)
            if len(feat_coords[0]) > 0:
                ax3.scatter(feat_coords[0], feat_coords[1], feat_coords[2], 
                          c=colors[feat_type], label=feature_names[feat_type],
                          alpha=0.6, marker='o')
        ax3.legend()
        ax3.set_title('Combined View')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'shape_ph4_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return shape_patches, ph4_patches

def generate_acp4_fingerprint(obmol) -> torch.Tensor:
    """Generate ACP4 fingerprint with proper error handling"""
    print("\n\n!!!! STARTING ACP4 FINGERPRINT GENERATION !!!!\n\n")
    try:
        # Get pharmacophore features
        ph4_coords, ph4_types = get_pharmacophore_features(obmol)
        smiles = obmol.write("smi").strip()  # Use OpenBabel's write method instead of RDKit
        
        if len(ph4_coords) == 0:
            print(f"No pharmacophore features found for: {smiles}")
            return torch.zeros(840, dtype=torch.float32)
        
        print(f"\nProcessing molecule: {smiles}")
        print(f"Found {len(ph4_coords)} pharmacophore features")
        
        # Generate PH4 file content
        ph4_content = []
        
        # First line: number of features and molecule identifier
        ph4_content.append(f"{len(ph4_coords)}:molecule")
        
        # Map feature types to ACP4 format
        feature_names = {
            0: 'ARO',  # Aromatic
            1: 'HBD',  # H-bond donor
            2: 'HBA',  # H-bond acceptor
            3: 'POS',  # Positive charge
            4: 'NEG',  # Negative charge
            5: 'HYD'   # Hydrophobic
        }
        
        # Write features in exact format from example
        for feat_idx in range(len(ph4_coords)):
            x, y, z = ph4_coords[feat_idx]
            feat_type = ph4_types[feat_idx].item()
            feat_name = feature_names[feat_type]
            ph4_content.append(f"{feat_name} {x:.5f} {y:.5f} {z:.5f}")
            print(f"Feature {feat_idx}: {feat_name} at ({x:.5f}, {y:.5f}, {z:.5f})")
        
        # Create temporary directory for ACP4 processing
        with tempfile.TemporaryDirectory() as tmpdir:
            ph4_path = os.path.join(tmpdir, "temp.ph4")
            
            # Write PH4 file
            with open(ph4_path, 'w') as f:
                f.write('\n'.join(ph4_content))
            
            # Run ACP4
            output_path = os.path.join(tmpdir, "temp.csv")
            cmd = [
                "./data/processed/all/acp4.exe",
                "-i", ph4_path,
                "-c", "5.0",
                "-dx", "0.5",
                "-o", output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    fp_str = f.read().strip()
                    if fp_str:
                        parts = fp_str.split()
                        if parts and parts[0] == '-1':  # Skip the -1 at start
                            parts = parts[1:]
                        fp = [float(x.split(':')[1]) if ':' in x else float(x) for x in parts]
                        return torch.tensor(fp, dtype=torch.float32)
            
            # If ACP4 failed, print everything immediately
            print("\n" + "#"*120)
            print("## ACP4 FAILURE REPORT")
            print("#"*120)
            print(f"\nSMILES: {smiles}")
            print("\nFeature coordinates:")
            print(ph4_coords)
            print("\nFeature types:")
            print(ph4_types)
            print("\nPH4 file content:")
            print("-" * 40)
            for line in ph4_content:
                print(line)
            print("-" * 40)
            print("\nACP4 command:")
            print(' '.join(cmd))
            print(f"\nReturn code: {result.returncode}")
            if result.stdout:
                print(f"Stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")
            print("#"*120 + "\n")
            
            # Also save to file for later analysis
            debug_dir = "failed_ph4_files"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_ph4_path = os.path.join(debug_dir, f"failed_{timestamp}.ph4")
            
            with open(failed_ph4_path, 'w') as f:
                f.write('\n'.join(ph4_content))
            print(f"PH4 file saved to: {failed_ph4_path}\n")
            
            return torch.zeros(840, dtype=torch.float32)
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return torch.zeros(840, dtype=torch.float32)
