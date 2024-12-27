from dataclasses import dataclass
import numpy as np

@dataclass
class ShapeFingerprint:
    shape: np.ndarray
    fingerprint: np.ndarray 