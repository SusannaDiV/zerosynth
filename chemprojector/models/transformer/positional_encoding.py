import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """Standard positional encoding for sequence data"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    """3D Spatial positional encoding for shape and ph4 patches"""
    def __init__(self, d_model: int, dropout: float = 0.1, grid_size: int = 7):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Calculate number of dimensions per coordinate (x,y,z)
        dim_per_coord = d_model // 3
        
        # Create position encodings for each spatial dimension
        pos_x = torch.arange(grid_size).unsqueeze(1)
        pos_y = torch.arange(grid_size).unsqueeze(1)
        pos_z = torch.arange(grid_size).unsqueeze(1)
        
        # Create frequency bands for each dimension
        div_term = torch.exp(torch.arange(0, dim_per_coord, 2) * (-math.log(10000.0) / dim_per_coord))
        
        # Initialize positional encoding tensor
        pe = torch.zeros(1, grid_size * grid_size * grid_size, d_model)
        
        # Generate all 3D coordinates
        coords = [(x, y, z) for x in range(grid_size) 
                           for y in range(grid_size) 
                           for z in range(grid_size)]
        
        # Fill positional encodings for each coordinate
        for idx, (x, y, z) in enumerate(coords):
            # X coordinate encoding
            pe[0, idx, 0:dim_per_coord:2] = torch.sin(x * div_term)
            pe[0, idx, 1:dim_per_coord:2] = torch.cos(x * div_term)
            
            # Y coordinate encoding
            pe[0, idx, dim_per_coord:2*dim_per_coord:2] = torch.sin(y * div_term)
            pe[0, idx, dim_per_coord+1:2*dim_per_coord:2] = torch.cos(y * div_term)
            
            # Z coordinate encoding
            pe[0, idx, 2*dim_per_coord:3*dim_per_coord:2] = torch.sin(z * div_term)
            pe[0, idx, 2*dim_per_coord+1:3*dim_per_coord:2] = torch.cos(z * div_term)
        
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, num_patches, embedding_dim]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
