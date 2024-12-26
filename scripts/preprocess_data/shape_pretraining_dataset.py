import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from math import ceil
from utils import get_atom_stamp, get_shape, get_shape_patches, get_atom_stamp_with_noise

class ShapePretrainingDataset(Dataset):
    """Dataset class to handle molecular shape data"""
    def __init__(self, 
                 data,
                 grid_resolution=1.0,
                 max_dist_stamp=3.0,
                 max_dist=10.0,
                 patch_size=3,
                 shape_noise_mu=0.0,
                 shape_noise_sigma=0.0):
        self.data = data
        self.grid_resolution = grid_resolution
        self.max_dist_stamp = max_dist_stamp
        self.max_dist = max_dist
        self.patch_size = patch_size
        self.shape_noise_mu = shape_noise_mu
        self.shape_noise_sigma = shape_noise_sigma
        self.box_size = ceil(2 * max_dist // grid_resolution + 1)
        self.atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        print(type(sample['mol']))
        # Get noisy atom stamp for data augmentation
        curr_atom_stamp = get_atom_stamp_with_noise(
            self.grid_resolution,
            self.max_dist_stamp,
            self.shape_noise_mu,
            self.shape_noise_sigma
        )
        # Get molecular shape
        curr_shape = get_shape(
            sample['mol'],
            curr_atom_stamp,
            self.grid_resolution,
            self.max_dist
        )
        
        # Process shape into patches
        print(f"curr_shape dimensions: {curr_shape.shape}")
        curr_shape_patches = get_shape_patches(curr_shape, self.patch_size)
        curr_shape_patches = curr_shape_patches.reshape(-1, self.patch_size**3)
        print("OK")

        return {
            'shape': torch.tensor(curr_shape, dtype=torch.long),
            'shape_patches': torch.tensor(curr_shape_patches, dtype=torch.float),
            'mol': sample['mol']  # Keep original molecule data if needed
        }

def collate_shapes(batch):
    """Custom collate function for the dataloader"""
    shapes = []
    shape_patches = []
    for item in batch:
        shapes.append(item['shape'])
        shape_patches.append(item['shape_patches'])

    # Stack all shapes and patches
    shapes = torch.stack(shapes)
    shape_patches = torch.stack(shape_patches)
    print(shape_patches.shape)
    return {
        'shape': shapes,  # [batch_size, box_size, box_size, box_size]
        'shape_patches': shape_patches,  # [batch_size, (box_size // patch_size)**3, patch_size**3]
    }

