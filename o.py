from chemprojector.data.common import _generate_shape_patches
import os

# Create visualization directory
os.makedirs("shape_visualization", exist_ok=True)

# Test with a simple molecule
smiles = "CC(=O)O"  # Acetic acid
patches = _generate_shape_patches(smiles)