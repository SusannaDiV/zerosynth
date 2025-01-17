import abc
from typing import TYPE_CHECKING

import torch
from torch import nn

from chemprojector.data.common import ProjectionBatch
from chemprojector.models.transformer.graph_transformer import GraphTransformer
from chemprojector.models.transformer.positional_encoding import PositionalEncoding

class BaseEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    if TYPE_CHECKING:

        def __call__(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]: ...


class SMILESEncoder(BaseEncoder):
    def __init__(
        self,
        num_token_types: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        pe_max_len: int,
    ):
        super().__init__()
        self._dim = d_model
        self.smiles_emb = nn.Embedding(num_token_types, d_model, padding_idx=0)
        self.pe_enc = PositionalEncoding(
            d_model=d_model,
            max_len=pe_max_len,
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if "smiles" not in batch:
            raise ValueError("smiles must be in batch")
        smiles = batch["smiles"]
        h = self.pe_enc(self.smiles_emb(smiles))
        padding_mask = smiles == 0  # the positions with the value of True will be ignored
        out = self.enc(h, src_key_padding_mask=padding_mask)
        return out, padding_mask


class GraphEncoder(BaseEncoder):
    def __init__(
        self,
        num_atom_classes: int,
        num_bond_classes: int,
        dim: int,
        depth: int,
        dim_head: int,
        edge_dim: int,
        heads: int,
        rel_pos_emb: bool,
        output_norm: bool,
    ):
        super().__init__()
        self._dim = dim
        self.atom_emb = nn.Embedding(num_atom_classes + 1, dim, padding_idx=0)
        self.bond_emb = nn.Embedding(num_bond_classes + 1, edge_dim, padding_idx=0)
        self.enc = GraphTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            edge_dim=edge_dim,
            heads=heads,
            rel_pos_emb=rel_pos_emb,
            output_norm=output_norm,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if "atoms" not in batch or "bonds" not in batch or "atom_padding_mask" not in batch:
            raise ValueError("atoms, bonds and atom_padding_mask must be in batch")
        atoms = batch["atoms"]
        bonds = batch["bonds"]
        atom_padding_mask = batch["atom_padding_mask"]

        atom_emb = self.atom_emb(atoms)
        bond_emb = self.bond_emb(bonds)
        node, _ = self.enc(nodes=atom_emb, edges=bond_emb, mask=atom_padding_mask)
        return node, atom_padding_mask

class ShapeEncoder(BaseEncoder):
    def __init__(
        self,
        patch_size: int = 3,
        d_model: int = 512,#d_model: int = 256, done to match the decoder
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_length: int = 3000
    ):
        super().__init__()
        self._dim = d_model
        self._patch_size = patch_size
        
        self._patch_ffn = nn.Sequential(
            nn.Linear(patch_size**3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self._pos_embed = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self._embed_dropout = nn.Dropout(0.1)
        self._transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self._norm = nn.LayerNorm(d_model)

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if "shape_patches" not in batch:
            raise ValueError("shape_patches must be in batch")
            
        shape_patches = batch["shape_patches"]
        bz, sl, _ = shape_patches.size()
        
        # Check if all patches are zero
        if torch.all(shape_patches == 0):
            print(f"WARNING: All shape patches are zero! Batch size: {bz}, Sequence length: {sl}")
        
        # Check percentage of zero elements
        #zero_percentage = (shape_patches == 0).float().mean().item() * 100
        #if zero_percentage > 99:  # If more than 99% are zeros
        #    print(f"WARNING: {zero_percentage:.2f}% of shape patch elements are zero! Batch size: {bz}, Sequence length: {sl}")
        
        x = self._patch_ffn(shape_patches)
        
        if sl > self._pos_embed.size(1):
            raise ValueError(f"Sequence length {sl} exceeds positional embedding size {self._pos_embed.size(1)}")
        
        pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
        x = x + self._pos_embed[:, :sl]
        x = self._embed_dropout(x)
        
        # Create padding mask (assuming no padding for now)
        padding_mask = torch.zeros((bz, sl), dtype=torch.bool, device=x.device)
        
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        x = self._transformer(x)
        x = self._norm(x)
        x = x.transpose(0, 1)  # Convert back to (batch, seq_len, d_model)
        
        return x, padding_mask 

def get_encoder(t: str, cfg) -> BaseEncoder:
    if t == "smiles":
        return SMILESEncoder(**cfg)
    elif t == "graph":
        return GraphEncoder(**cfg)
    elif t == "shape":
        print("RRUNING SHAPE ENCODER")
        return ShapeEncoder(**cfg)
    else:
        raise ValueError(f"Unknown encoder type: {t}")
