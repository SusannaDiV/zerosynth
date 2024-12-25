import numpy as np
from typing import List, Tuple
import pathlib

class Ph4Fingerprint:
    # Define feature types and their corresponding grid dimensions
    FEATURE_TYPES = ['ARO', 'HBD', 'HBA', 'POS', 'NEG', 'HYD']
    GRID_SIZE = 20  # Number of bins in each dimension
    GRID_RANGE = (-20, 40)  # Spatial range in Angstroms
    
    @staticmethod
    def _parse_ph4_file(ph4_path: pathlib.Path) -> List[Tuple[str, np.ndarray]]:
        features = []
        with open(ph4_path) as f:
            next(f)  # Skip header line
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    feat_type = parts[0]
                    coords = np.array([float(x) for x in parts[1:4]])
                    features.append((feat_type, coords))
        return features
    
    @staticmethod
    def _get_grid_index(coord: float, min_val: float, max_val: float, n_bins: int) -> int:
        bin_size = (max_val - min_val) / n_bins
        idx = int((coord - min_val) / bin_size)
        return max(0, min(n_bins - 1, idx))
    
    @classmethod
    def compute_fingerprint(cls, ph4_path: pathlib.Path) -> np.ndarray:
        # Initialize 4D array (feature_type, x, y, z)
        fp = np.zeros((len(cls.FEATURE_TYPES), cls.GRID_SIZE, cls.GRID_SIZE, cls.GRID_SIZE))
        
        # Parse features from ph4 file
        features = cls._parse_ph4_file(ph4_path)
        
        # Populate the grid
        for feat_type, coords in features:
            if feat_type in cls.FEATURE_TYPES:
                feat_idx = cls.FEATURE_TYPES.index(feat_type)
                x_idx = cls._get_grid_index(coords[0], cls.GRID_RANGE[0], cls.GRID_RANGE[1], cls.GRID_SIZE)
                y_idx = cls._get_grid_index(coords[1], cls.GRID_RANGE[0], cls.GRID_RANGE[1], cls.GRID_SIZE)
                z_idx = cls._get_grid_index(coords[2], cls.GRID_RANGE[0], cls.GRID_RANGE[1], cls.GRID_SIZE)
                fp[feat_idx, x_idx, y_idx, z_idx] += 1
        
        # Flatten to 1D array
        return fp.flatten()
    
    @classmethod
    def dim(cls) -> int:
        """Return dimension of the flattened fingerprint"""
        return len(cls.FEATURE_TYPES) * cls.GRID_SIZE ** 3 