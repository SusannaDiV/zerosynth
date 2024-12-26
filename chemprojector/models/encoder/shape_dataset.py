import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import get_shape_patches

class ShapePretrainingDataset(Dataset):
    def __init__(self, data, grid_resolution, max_dist_stamp, max_dist, patch_size):
        self.data = data
        self.grid_resolution = grid_resolution
        self.max_dist_stamp = max_dist_stamp
        self.max_dist = max_dist
        self.patch_size = patch_size

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