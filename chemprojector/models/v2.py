import abc
from typing import TYPE_CHECKING

import torch
from torch import nn

from chemprojector.data.common import ProjectionBatch
from chemprojector.models.transformer.graph_transformer import GraphTransformer
from chemprojector.models.transformer.positional_encoding import PositionalEncoding, SpatialPositionalEncoding

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
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        grid_size: int = 7,  # 7x7x7 grid = 343 patches
        dropout: float = 0.1,
        max_seq_length: int = 5000,  # Keep for compatibility
        fusion_type: str = "cross_attn"  # Options: "concat", "gated", "cross_attn", "bilinear"
    ):
        super().__init__()
        self._dim = d_model
        self._patch_size = patch_size
        self.fusion_type = fusion_type
        
        # Separate FFNs for shape and ph4 patches
        self._shape_ffn = nn.Sequential(
            nn.Linear(patch_size**3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self._ph4_ffn = nn.Sequential(
            nn.Linear(patch_size**3 * 6, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Rotary embeddings for better position understanding
        self.pos_emb = RotaryEmbedding(d_model // nhead)
        
        # Feature fusion modules
        if fusion_type == "concat":
            self._fusion_layer = PreNorm(d_model * 2, nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            ))
        elif fusion_type == "gated":
            self._fusion_layer = PreNorm(d_model * 2, GatedFusion(d_model))
        elif fusion_type == "cross_attn":
            self._fusion_layer = PreNorm(d_model, CrossAttentionFusion(d_model, nhead, dropout))
        elif fusion_type == "bilinear":
            self._fusion_layer = PreNorm(d_model * 2, BilinearFusion(d_model))
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Transformer layers with PreNorm and GatedResidual
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                )),
                GatedResidual(d_model)
            ]))
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if "shape_patches" not in batch or "ph4_patches" not in batch:
            raise ValueError("shape_patches and ph4_patches must be in batch")
            
        shape_patches = batch["shape_patches"]
        ph4_patches = batch["ph4_patches"]
        bz, sl, _ = shape_patches.size()
        
        # Process patches through FFNs
        x_shape = self._shape_ffn(shape_patches)
        x_ph4 = self._ph4_ffn(ph4_patches)
        
        # Apply rotary embeddings
        # Note: rotary embeddings are applied to queries and keys in attention
        if self.fusion_type == "cross_attn":
            # For cross attention, apply to both shape and ph4
            x_shape = self.pos_emb.rotate_queries_or_keys(x_shape)
            x_ph4 = self.pos_emb.rotate_queries_or_keys(x_ph4)
        
        # Fuse features using the selected method
        if self.fusion_type == "concat":
            x = self._fusion_layer(torch.cat([x_shape, x_ph4], dim=-1))
        elif self.fusion_type in ["gated", "bilinear"]:
            x = self._fusion_layer(x_shape, x_ph4)
        elif self.fusion_type == "cross_attn":
            x = self._fusion_layer(x_shape, x_ph4)
        
        # Process through transformer layers with rotary embeddings
        padding_mask = torch.zeros((bz, sl), dtype=torch.bool, device=x.device)
        
        for attn, residual in self.layers:
            # Apply rotary embeddings within attention
            queries = self.pos_emb.rotate_queries_or_keys(x)
            keys = self.pos_emb.rotate_queries_or_keys(x)
            
            # Apply attention with PreNorm
            attended, _ = attn(queries, keys, x, key_padding_mask=padding_mask)
            
            # Apply gated residual
            x = residual(attended, x)
        
        x = self.norm(x)
        return x, padding_mask

# PreNorm wrapper
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# Gated Residual
class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

# Different fusion methods
class GatedFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x_shape: torch.Tensor, x_ph4: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_shape, x_ph4], dim=-1)
        gate = self.gate(combined)
        fused = self.fusion(combined)
        return gate * fused

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x_shape: torch.Tensor, x_ph4: torch.Tensor) -> torch.Tensor:
        attn1, _ = self.cross_attn1(x_shape, x_ph4, x_ph4)
        attn2, _ = self.cross_attn2(x_ph4, x_shape, x_shape)
        attn1 = self.norm1(attn1 + x_shape)
        attn2 = self.norm2(attn2 + x_ph4)
        return self.fusion(torch.cat([attn1, attn2], dim=-1))

class BilinearFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x_shape: torch.Tensor, x_ph4: torch.Tensor) -> torch.Tensor:
        bilinear = self.bilinear(x_shape, x_ph4)
        return self.norm(bilinear + x_shape + x_ph4)

def get_encoder(t: str, cfg) -> BaseEncoder:
    if t == "smiles":
        return SMILESEncoder(**cfg)
    elif t == "graph":
        return GraphEncoder(**cfg)
    elif t == "shape":
        print("RUNNING SHAPE ENCODER")
        return ShapeEncoder(**cfg)
    else:
        raise ValueError(f"Unknown encoder type: {t}")
