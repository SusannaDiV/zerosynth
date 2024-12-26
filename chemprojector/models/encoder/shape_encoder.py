import torch
import torch.nn as nn

class ShapePretrainingEncoder(nn.Module):
    def __init__(self, patch_size, d_model, num_layers, nhead, max_seq_length=3000):
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
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self._norm = nn.LayerNorm(d_model)

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, src):
        bz, sl, _ = src.size()
        x = self._patch_ffn(src)
        pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
        if sl > self._pos_embed.size(1):
            raise ValueError(f"Sequence length {sl} exceeds positional embedding size {self._pos_embed.size(1)}")
        #x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        x = x + self._pos_embed[:, :sl]
        x = self._embed_dropout(x)
        x = self._transformer(x)
        x = self._norm(x)
        x = self._norm(x)
        # Create padding mask (all False since shape patches are fixed size)
        padding_mask = torch.zeros((bz, sl), dtype=torch.bool, device=x.device)
        
        return x, padding_mask