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
    #Remove it here likley
    try:
        seq_idx = mol_seq.index(product)
        mol_idx = mol_idx_seq[seq_idx]
    except ValueError:
        ##print(f"Warning: Product molecule not found in sequence, using last molecule")
        mol_idx = mol_idx_seq[-1]

    # Always keep tensors on CPU in worker processes
    if mol_idx in fpindex._shape_patches:
        shape_patches = fpindex._shape_patches[mol_idx].cpu()
        print("nonzero common")

    else:
        ##print(f"No shape patches found for molecule {mol_idx}")
        shape_patches = torch.zeros((343, 27), dtype=torch.float32, device='cpu')
# remove

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
