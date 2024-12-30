import numpy as np
from typing import List, Tuple, Dict
from rdkit import Chem
from .allinacp4_ph4 import (
    find_ARO, find_HBD, find_HBA, 
    find_POS, find_NEG, find_HYD,
    FEATURE_TYPES
)

class GaussianPH4Generator:
    def __init__(
        self, 
        sigma: float = 1.0,
        feature_types: List[str] = FEATURE_TYPES,
        distance_cutoff: float = 12.0,  # Å
        normalize: bool = True
    ):
        self.sigma = sigma
        self.feature_types = feature_types
        self.distance_cutoff = distance_cutoff
        self.normalize = normalize
        
        # Map feature types to their extraction functions
        self.feature_extractors = {
            'ARO': find_ARO,
            'HBD': find_HBD,
            'HBA': find_HBA,
            'POS': find_POS,
            'NEG': find_NEG,
            'HYD': find_HYD
        }

    def gaussian_kernel(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Compute Gaussian kernel between two 3D coordinates"""
        dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
        if dist > self.distance_cutoff:
            return 0.0
        return np.exp(-dist**2 / (2 * self.sigma**2))

    def compute_feature_interactions(
        self, 
        features1: List[Tuple[float, float, float]], 
        features2: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """Compute all pairwise Gaussian interactions between two feature sets"""
        if not features1 or not features2:
            return np.zeros(1)
            
        interactions = []
        for f1 in features1:
            for f2 in features2:
                interactions.append(self.gaussian_kernel(f1, f2))
        return np.array(interactions)

    def get_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Generate the complete Gaussian invariant PH4 fingerprint"""
        # Extract all features
        features_dict = {}
        for feat_type in self.feature_types:
            features = self.feature_extractors[feat_type](mol)
            features_dict[feat_type] = features

        # Compute all pairwise feature type interactions
        fingerprint_parts = []
        
        for i, type1 in enumerate(self.feature_types):
            for j, type2 in enumerate(self.feature_types[i:], i):
                interactions = self.compute_feature_interactions(
                    features_dict[type1],
                    features_dict[type2]
                )
                
                # Compute statistical moments of interactions
                if len(interactions) > 0:
                    moments = [
                        np.mean(interactions),
                        np.std(interactions) if len(interactions) > 1 else 0,
                        np.max(interactions) if len(interactions) > 0 else 0
                    ]
                else:
                    moments = [0, 0, 0]
                    
                fingerprint_parts.extend(moments)

        fp = np.array(fingerprint_parts, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize and np.any(fp):
            fp = fp / np.linalg.norm(fp)
            
        return fp

    @property
    def dim(self) -> int:
        """Calculate fingerprint dimension"""
        n_types = len(self.feature_types)
        n_pairs = (n_types * (n_types + 1)) // 2  # Number of unique feature type pairs
        n_moments = 3  # mean, std, max for each interaction type
        return n_pairs * n_moments 
    
'''
Config:
chem:
  fp_option:
    type: gaussian_ph4
    gaussian_sigma: 1.0
    gaussian_distance_cutoff: 12.0
    gaussian_normalize: true
  
model:
  decoder:
    fingerprint_dim: 63  # For 6 feature types: (6 * 7/2) pairs * 3 moments
'''