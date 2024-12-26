import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import get_shape_patches

class ShapePretrainingDataset(Dataset):
    def __init__(self, data, grid_resolution, max_dist_stamp, max_dist, patch_size):
        if not isinstance(data, (list, np.ndarray)):
            raise TypeError(f"Data must be list or numpy array, got {type(data)}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
            
        self._validate_numeric_param("grid_resolution", grid_resolution, min_val=0.0)
        self._validate_numeric_param("max_dist_stamp", max_dist_stamp, min_val=0.0)
        self._validate_numeric_param("max_dist", max_dist, min_val=0.0)
        self._validate_numeric_param("patch_size", patch_size, min_val=1)
        
        self.data = data
        self.grid_resolution = grid_resolution
        self.max_dist_stamp = max_dist_stamp
        self.max_dist = max_dist
        self.patch_size = patch_size

    def _validate_numeric_param(self, name, value, min_val=None, max_val=None):
        """Validates numeric parameters."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(value)}")
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        shape = item['mol']  # Assuming shape is stored in 'mol' key
        
        # Convert shape to patches
        shape_patches = get_shape_patches(shape, self.patch_size)
        shape_patches = shape_patches.reshape(-1, self.patch_size**3)
        
        return {
            'shape': torch.FloatTensor(shape),
            'shape_patches': torch.FloatTensor(shape_patches)
        } 