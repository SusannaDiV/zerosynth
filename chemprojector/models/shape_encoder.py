import torch
import torch.nn as nn
from .shape_encodings import get_grid_position_encoding, get_rotation_encoding

class ShapeEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 3,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_length: int = 3000,
        grid_resolution: float = 0.5,
        max_dist: float = 15.0,
        num_rotation_angles: int = 24,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding
        self.patch_ffn = nn.Sequential(
            nn.Linear(patch_size**3, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        # Position encoding
        self.register_buffer(
            "position_encoding",
            get_grid_position_encoding(30, patch_size, grid_resolution, max_dist)  # 30 is typical shape size
        )
        self.pos_encoder = nn.Linear(3, d_model)
        
        # Rotation encoding
        self.register_buffer(
            "rotation_encoding",
            get_rotation_encoding(num_rotation_angles)
        )
        self.rot_encoder = nn.Linear(9, d_model)  # 9 = 3x3 flattened rotation matrix
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, batch):
        if "shape_patches" not in batch:
            raise ValueError("shape_patches must be in batch")
        
        shape_patches = batch["shape_patches"]
        bz, sl, _ = shape_patches.size()
        
        # Patch embedding
        x = self.patch_ffn(shape_patches)  # [B, N, D]
        
        # Add position encoding
        pos_enc = self.pos_encoder(self.position_encoding[:sl])  # [N, D]
        x = x + pos_enc.unsqueeze(0)  # [B, N, D]
        
        # Add rotation encoding (simplified version - can be made more sophisticated)
        rot_enc = self.rot_encoder(self.rotation_encoding.view(-1, 9))  # [72, D] for num_angles=24
        x = x + rot_enc.mean(0, keepdim=True).unsqueeze(0)  # [B, N, D]
        
        # Create padding mask (assuming no padding for now)
        padding_mask = torch.zeros((bz, sl), dtype=torch.bool, device=x.device)
        
        # Apply transformer
        x = x.transpose(0, 1)  # [N, B, D]
        x = self.transformer(x)
        x = self.norm(x)
        x = x.transpose(0, 1)  # [B, N, D]
        
        return x