import enum
from collections.abc import Sequence
from typing import TypedDict

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


def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    mol_idx_seq: Sequence[int | None],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    fpindex: FingerprintIndex,
    encoder_type: str = "shape",
    device: torch.device = None,
):
    # Try direct index first
    if product in mol_seq:
        seq_idx = mol_seq.index(product)
    else:
        # If direct lookup fails, try SMILES comparison
        seq_idx = None
        product_smiles = product.smiles
        for i, mol in enumerate(mol_seq):
            if mol.smiles == product_smiles:
                seq_idx = i
                break
    
    if seq_idx is None:
        mol_idx = None
    else:
        mol_idx = mol_idx_seq[seq_idx]

    if mol_idx is None or mol_idx not in fpindex._shape_patches:
        try:
            # Inlined compute_shape_patches function
            with torch.device('cpu'):
                # Generate 3D conformer with explicit checks
                rdmol = Chem.AddHs(product._rdmol)
                if rdmol is None:
                    raise ValueError("Failed to add hydrogens")
                
                # Try multiple embedding methods in sequence
                embed_success = False
                
                # 1. Try ETKDGv3 with default params
                embed_params = AllChem.ETKDGv3()
                embed_params.randomSeed = 42
                embed_params.maxIterations = 1000
                embed_result = AllChem.EmbedMolecule(rdmol, embed_params)
                
                if embed_result == -1:
                    # 2. Try with random coords
                    embed_params.useRandomCoords = True
                    embed_result = AllChem.EmbedMolecule(rdmol, embed_params)
                    
                    if embed_result == -1:
                        # 3. Try ETDKG with different parameters
                        embed_params = AllChem.ETKDG()
                        embed_params.randomSeed = 42
                        embed_params.useBasicKnowledge = True
                        embed_params.enforceChirality = False
                        embed_result = AllChem.EmbedMolecule(rdmol, embed_params)
                        
                        if embed_result == -1:
                            # 4. Try distance geometry with basic parameters
                            embed_params = AllChem.srETKDGv3()
                            embed_params.randomSeed = 42
                            embed_result = AllChem.EmbedMolecule(rdmol, embed_params)
                            
                            if embed_result == -1:
                                raise ValueError("Failed to embed molecule after multiple attempts")
                
                # Try MMFF optimization first, fall back to UFF
                optimize_result = AllChem.MMFFOptimizeMolecule(rdmol)
                if optimize_result == -1:
                    optimize_result = AllChem.UFFOptimizeMolecule(rdmol)
                
                rdmol = Chem.RemoveHs(rdmol)
                if rdmol is None:
                    raise ValueError("Failed to remove hydrogens")
                
                # Create new molecule with conformer properly copied
                new_rdmol = Chem.Mol(rdmol)
                conf = rdmol.GetConformer()
                new_rdmol.AddConformer(conf)
                
                # Create cavity from the new molecule with conformer
                cavity = Chem.Mol(new_rdmol)
                if cavity is None:
                    raise ValueError("Failed to create cavity")
                
                # Verify cavity has conformer
                if cavity.GetNumConformers() == 0:
                    raise ValueError("Cavity has no conformer")
                
                # Use same parameters as in process_single_rotation
                resolution = 0.5
                box_size = 15
                atom_stamp = get_atom_stamp(grid_resolution=resolution, max_dist=4.0)
                
                # Use memory-safe shape computation
                curr_cavity_shape = get_shape(
                    mol=cavity,
                    atom_stamp=atom_stamp,
                    grid_resolution=resolution,
                    max_dist=box_size
                )
                
                if curr_cavity_shape is None:
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
                
                # Convert to tensor on CPU first
                shape_patches = torch.from_numpy(shape_patches).to(torch.float32)
                #print("Computing shape patches for new product")

        except Exception as e:
            ### Check why error codes 5 and 34 based on fpindex 
            # print(f"WARNING: Failed to compute shape patches for new product: {e}")
            # Return zero tensor on CPU
            shape_patches = torch.zeros((343, 27), dtype=torch.float32, device='cpu')
    else:
        shape_patches = fpindex._shape_patches[mol_idx].cpu()
        print("nonzero common")

    stack_feats = featurize_stack_actions(
        mol_idx_seq=mol_idx_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
        fpindex=fpindex,
    )

    # Keep everything on CPU
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
