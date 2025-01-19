from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


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


def collate_shape_patches(features: list[torch.Tensor], max_size: int | None = None) -> torch.Tensor:
    normalized_features = []
    for f in features:
        if f.shape[0] == 1:
            f = f.repeat(343, 1)
        elif f.shape[0] != 343:
            raise ValueError(f"Unexpected shape patch size: {f.shape}. Expected (343, 27) or (1, 27)")
        normalized_features.append(f)
    
    return torch.stack(normalized_features, dim=0)


def collate_ph4_patches(features: list[torch.Tensor], max_size: int | None = None) -> torch.Tensor:
    normalized_features = []
    for f in features:
        if isinstance(f, tuple):
            ph4_patch = f[1]  # Extract ph4 patches from tuple
        else:
            ph4_patch = f  # Use the tensor directly
            
        # Validate and normalize the shape
        if ph4_patch.shape[0] == 1:
            ph4_patch = ph4_patch.repeat(343, 1)
        elif ph4_patch.shape[0] != 343:
            raise ValueError(f"Unexpected ph4 patch size: {ph4_patch.shape}. Expected (343, 162) or (1, 162)")
        
        # Ensure float32 type
        ph4_patch = ph4_patch.to(torch.float32)
        normalized_features.append(ph4_patch)
    
    # Stack the features and return
    return torch.stack(normalized_features, dim=0)


def collate_acp4_fp(features: list[torch.Tensor], max_size: int | None = None) -> torch.Tensor:
    """Collate ACP4 fingerprints.
    
    Args:
        features: List of ACP4 fingerprint tensors, each of shape [840]
        max_size: Ignored, kept for compatibility with other collate functions
        
    Returns:
        Batched tensor of shape [batch_size, 840]
    """
    normalized_features = []
    for f in features:
        if isinstance(f, tuple):
            acp4_fp = f[2]  # Extract ACP4 fp from tuple if needed
        else:
            acp4_fp = f
            
        # Validate shape
        if acp4_fp.shape != torch.Size([840]):
            raise ValueError(f"Unexpected ACP4 fingerprint size: {acp4_fp.shape}. Expected (840,)")
        
        # Ensure float32 type
        acp4_fp = acp4_fp.to(torch.float32)
        normalized_features.append(acp4_fp)
    
    # Stack the features and return
    return torch.stack(normalized_features, dim=0)


def apply_collate(
    spec: Mapping[str, Callable[[Sequence[torch.Tensor], int], torch.Tensor]],
    data_list: Sequence[dict[str, torch.Tensor]],
    max_size: int,
) -> dict[str, torch.Tensor]:
    transpose = {k: [d[k] for d in data_list] for k in spec.keys()}
    batch = {k: spec[k](transpose[k], max_size) for k in spec.keys()}
    return batch
