import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapePretrainingEncoder(nn.Module):
    def __init__(self, patch_size, d_model, num_layers, nhead, max_seq_length=3000):
        super().__init__()
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

    def forward(self, src):
        bz, sl, _ = src.size()
        x = self._patch_ffn(src)
        
        if sl > self._pos_embed.size(1):
            raise ValueError(f"Sequence length {sl} exceeds positional embedding size {self._pos_embed.size(1)}")
        
        pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
        x = x + self._pos_embed[:, :sl]
        x = self._embed_dropout(x)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        x = self._transformer(x)
        x = self._norm(x)
        return x
'''
import torch
import torch.nn as nn

class TransformerEncoderBase(nn.Module):
    """Base class to match Bycha's TransformerEncoder functionality""" # TODO: check if this is correct
    def __init__(self, d_model, nhead, num_layers, return_seed=False):
        super().__init__()
        self._d_model = d_model
        self._return_seed = return_seed
        self._embed_scale = None
        self._pos_embed = None
        self._embed_norm = None
        self._embed_dropout = nn.Dropout(0.1)
        self._layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead) 
            for _ in range(num_layers)
        ])
        self._norm = nn.LayerNorm(d_model)
    
    def build(self, embed, special_tokens):
        """Initialize embeddings and normalization"""
        self._embed_scale = nn.Parameter(torch.ones(1))
        self._pos_embed = nn.Embedding(1000, self._d_model)  # Assuming max length of 1000
        self._embed_norm = nn.LayerNorm(self._d_model)


class ShapePretrainingEncoder(TransformerEncoderBase):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
    
    def build(self, embed, special_tokens):
        super().build(embed, special_tokens)
        # FFN replacement with equivalent PyTorch layers
        self._patch_ffn = nn.Sequential(
            nn.Linear(self._patch_size**3, self._d_model),
            nn.ReLU(),
            nn.Linear(self._d_model, self._d_model)
        )
    
    def forward(self, src):
        bz, sl = src.size(0), src.size(1)
        
        x = self._patch_ffn(src)
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
            x = x + self._pos_embed(pos)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        #src_padding_mask = torch.zeros((bz, sl), dtype=torch.bool).to(x.device)
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x)#, src_key_padding_mask=src_padding_mask)
        
        if self._norm is not None:
            x = self._norm(x)
        
        if self._return_seed:
            encoder_out = x[1:]#, src_padding_mask[:, 1:], x[0]
        else:
            encoder_out = x#, src_padding_mask

        return encoder_out
'''