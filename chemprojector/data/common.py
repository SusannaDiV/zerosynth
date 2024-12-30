import enum
from collections.abc import Sequence
from typing import TypedDict

import torch
import numpy as np

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.mol import Molecule
from chemprojector.chem.reaction import Reaction
from chemprojector.chem.stack import Stack
from chemprojector.utils.image import draw_text, make_grid


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
    shape: torch.Tensor
    shape_patches: torch.Tensor
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Molecule]
    rxn_seq: Sequence[Reaction | None]


class ProjectionBatch(TypedDict, total=False):
    # Encoder
    '''
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    '''
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Sequence[Molecule]]
    rxn_seq: Sequence[Sequence[Reaction | None]]
    # Shape encoder fields
    shape: torch.Tensor  # [batch_size, box_size, box_size, box_size]
    shape_patches: torch.Tensor  # [batch_size, num_patches, patch_size**3]
    shape_padding_mask: torch.Tensor  # [batch_size, num_patches]


def featurize_stack_actions(
    mol_idx_seq: Sequence[int | None],
    rxn_idx_seq: Sequence[int | None],
    end_token: bool,
    fpindex: FingerprintIndex,
    shape_data: list | None = None,
    encoder_type: str = "shape"
) -> dict[str, torch.Tensor]:
    seq_len = len(mol_idx_seq) + 1
    if end_token:
        seq_len += 1
    fp_dim = fpindex.fp_option.dim
    feats = {
        "token_types": torch.zeros([seq_len], dtype=torch.long),
        "rxn_indices": torch.zeros([seq_len], dtype=torch.long),
        "reactant_fps": torch.zeros([seq_len, fp_dim], dtype=torch.float32),
        "token_padding_mask": torch.zeros([seq_len], dtype=torch.bool),
    }
    
    if encoder_type == "shape" and shape_data is not None:
        feats["shape_seq"] = []
        feats["shape_patches_seq"] = []
    
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
            
            # Only process shape data if using shape encoder, might be useless, who knows
            if encoder_type == "shape" and shape_data is not None:
                shape_item = shape_data[mol_idx]
                feats["shape_seq"].append(torch.tensor(shape_item['shape'], dtype=torch.float))
                feats["shape_patches_seq"].append(create_patches(shape_item['shape']))
    
    return feats


def featurize_stack(
    stack: Stack, 
    end_token: bool, 
    fpindex: FingerprintIndex, 
    shape_data: list | None = None,
    encoder_type: str = "shape"
) -> dict[str, torch.Tensor]:
    return featurize_stack_actions(
        mol_idx_seq=stack.get_mol_idx_seq(),
        rxn_idx_seq=stack.get_rxn_idx_seq(),
        end_token=end_token,
        fpindex=fpindex,
        shape_data=shape_data,
        encoder_type=encoder_type
    )


def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    mol_idx_seq: Sequence[int | None],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    fpindex: FingerprintIndex,
    shape_data: list | None = None,
    shape: torch.Tensor = None,
    shape_patches: torch.Tensor = None,
    encoder_type: str = "shape"
):
    stack_feats = featurize_stack_actions(
        mol_idx_seq=mol_idx_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
        fpindex=fpindex,
        shape_data=shape_data,
        encoder_type=encoder_type
    )

    data: "ProjectionData" = {
        "mol_seq": mol_seq,
        "rxn_seq": rxn_seq,
        "token_types": stack_feats["token_types"],
        "rxn_indices": stack_feats["rxn_indices"],
        "reactant_fps": stack_feats["reactant_fps"],
        "token_padding_mask": stack_feats["token_padding_mask"],
    }
    
    # Only add structural data if not using shape encoder
    '''
    if encoder_type == "graph":
        print("SHAPE ENCODER NOT USED")
        atom_f, bond_f = product.featurize_simple()
        data.update({
            "atoms": atom_f,
            "bonds": bond_f,
            "smiles": product.tokenize_csmiles(),
            "atom_padding_mask": torch.zeros([atom_f.size(0)], dtype=torch.bool),
        })
        '''
    if shape is not None:
        data["shape"] = shape
    if shape_patches is not None:
        data["shape_patches"] = shape_patches
        
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
