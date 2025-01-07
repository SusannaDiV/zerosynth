import numpy as np
from math import ceil
import torch

def get_grid_position_encoding(shape_size: int, patch_size: int, grid_resolution: float = 0.5, max_dist: float = 15.0):
    """
    Create position encoding for each patch based on its position in the 3D grid
    
    Args:
        shape_size: Size of the shape grid (assumed cubic)
        patch_size: Size of each patch (assumed cubic)
        grid_resolution: Resolution of the grid
        max_dist: Maximum distance in Angstroms
    """
    num_patches = shape_size // patch_size
    positions = []
    
    # Calculate center positions for each patch
    for i in range(num_patches):
        for j in range(num_patches):
            for k in range(num_patches):
                # Calculate center of patch in grid coordinates
                center_x = (i + 0.5) * patch_size
                center_y = (j + 0.5) * patch_size
                center_z = (k + 0.5) * patch_size
                
                # Convert to real space coordinates
                real_x = (center_x - shape_size/2) * grid_resolution
                real_y = (center_y - shape_size/2) * grid_resolution
                real_z = (center_z - shape_size/2) * grid_resolution
                
                positions.append([real_x, real_y, real_z])
    
    return torch.tensor(positions, dtype=torch.float32)

def get_rotation_encoding(num_angles: int = 24):
    """
    Create rotation encoding using discretized angles
    
    Args:
        num_angles: Number of discrete rotation angles to use
    """
    angles = torch.linspace(0, 2*np.pi, num_angles)
    
    # Create rotation matrices for each angle around x, y, and z axes
    rot_x = torch.stack([torch.tensor([
        [1., 0., 0.],
        [0., torch.cos(a), -torch.sin(a)],
        [0., torch.sin(a), torch.cos(a)]
    ]) for a in angles])
    
    rot_y = torch.stack([torch.tensor([
        [torch.cos(a), 0., torch.sin(a)],
        [0., 1., 0.],
        [-torch.sin(a), 0., torch.cos(a)]
    ]) for a in angles])
    
    rot_z = torch.stack([torch.tensor([
        [torch.cos(a), -torch.sin(a), 0.],
        [torch.sin(a), torch.cos(a), 0.],
        [0., 0., 1.]
    ]) for a in angles])
    
    return torch.cat([rot_x, rot_y, rot_z], dim=0)  # 3*num_angles x 3 x 3 