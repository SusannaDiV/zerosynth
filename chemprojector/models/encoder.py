import abc
from typing import TYPE_CHECKING
import torch
from torch import nn
from chemprojector.data.common import ProjectionBatch

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
        x = x + self._pos_embed[:, :sl]
        x = self._embed_dropout(x)
        x = self._transformer(x)
        x = self._norm(x)
        x = self._norm(x)
        padding_mask = torch.zeros((bz, sl), dtype=torch.bool, device=x.device)
        return x, padding_mask

class BaseEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    if TYPE_CHECKING:
        def __call__(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]: ...

class ShapeEncoder(BaseEncoder):
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        max_seq_length: int = 3000
    ):
        super().__init__()
        self._dim = d_model
        self.shape_encoder = ShapePretrainingEncoder(
            patch_size=patch_size,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            max_seq_length=max_seq_length
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, shapes, shape_patches) -> tuple[torch.Tensor, torch.Tensor]:
        out, padding_mask = self.shape_encoder(shape_patches)
        return out, padding_mask

def get_encoder(t: str, cfg) -> BaseEncoder:
    if t == "shape":
        encoder_cfg = {k: v for k, v in cfg.items() if k != "encoder_type"}
        return ShapeEncoder(**encoder_cfg)
    else:
        raise ValueError("Only shape encoder is supported")
