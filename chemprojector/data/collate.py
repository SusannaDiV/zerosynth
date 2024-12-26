from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict, cast

import torch
import torch.nn.functional as F

from .common import ShapeBatch

__all__ = ['ShapeCollater', 'collate_tokens', 'collate_2d_tokens', 'collate_1d_features', 'collate_padding_masks']


def collate_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [F.pad(f, pad=[0, max_size - f.size(-1)], mode="constant", value=0) for f in features]
    return torch.stack(features_padded, dim=0)


def collate_2d_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=[0, max_size - f.size(-1), 0, max_size - f.size(-2)], mode="constant", value=0) for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_1d_features(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [F.pad(f, pad=[0, 0, 0, max_size - f.size(-2)], mode="constant", value=0) for f in features]
    return torch.stack(features_padded, dim=0)


def collate_2d_features(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=[0, 0, 0, max_size - f.size(-2), 0, max_size - f.size(-3)], mode="constant", value=0)
        for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_padding_masks(masks: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    masks_padded = [F.pad(m, pad=[0, max_size - m.size(-1)], mode="constant", value=True) for m in masks]
    return torch.stack(masks_padded, dim=0)


def apply_collate(
    spec: Mapping[str, Callable[[Sequence[torch.Tensor], int], torch.Tensor]],
    data_list: Sequence[dict[str, torch.Tensor]],
    max_size: int,
) -> dict[str, torch.Tensor]:
    transpose = {k: [d[k] for d in data_list] for k in spec.keys()}
    batch = {k: spec[k](transpose[k], max_size) for k in spec.keys()}
    return batch


def extract_patches(shape, patch_size):
    """Extract non-overlapping patches from 3D shape using view"""
    # Check if dimensions are divisible by patch_size
    if any(dim % patch_size != 0 for dim in shape.shape):
        raise ValueError(f"Shape dimensions {shape.shape} must be divisible by patch_size {patch_size}")
    
    # Unfold each dimension
    patches = shape.unfold(0, patch_size, patch_size)\
                  .unfold(1, patch_size, patch_size)\
                  .unfold(2, patch_size, patch_size)
    
    # Reshape to match original format: (N_patches, patch_size^3)
    return patches.reshape(-1, patch_size**3)


class ShapeCollater:
    def __init__(self, patch_size: int = 3):
        self.patch_size = patch_size

    def __call__(self, batch) -> ShapeBatch:
        shapes = [item["shape"] for item in batch]
        shape_patches = [extract_patches(shape, self.patch_size) for shape in shapes]
        
        return {
            "shape_patches": torch.stack(shape_patches),
            "shapes": torch.stack([s.clone().detach() for s in shapes])
        }
