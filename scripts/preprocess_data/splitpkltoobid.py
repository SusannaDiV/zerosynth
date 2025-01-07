import pickle
import pathlib

# Paths to the original and new files
original_fpindex_path = 'data/processed/all/fpindex_e3fp.pkl'
shapes_path = 'data/processed/all/shapes.pkl'
fpindex_without_shapes_path = 'data/processed/all/fpindex_without_shapes.pkl'

# Load the original FingerprintIndex
with open(original_fpindex_path, 'rb') as f:
    fpindex = pickle.load(f)

# Extract shapes and shape patches
shapes = fpindex._shapes
shape_patches = fpindex._shape_patches

# Save shapes and shape patches to a new file
with open(shapes_path, 'wb') as f:
    pickle.dump((shapes, shape_patches), f)

# Remove shapes and shape patches from the original object
fpindex._shapes = {}
fpindex._shape_patches = {}

# Save the modified FingerprintIndex without shapes
with open(fpindex_without_shapes_path, 'wb') as f:
    pickle.dump(fpindex, f)

print(f"Shapes and shape patches saved to {shapes_path}")
print(f"FingerprintIndex without shapes saved to {fpindex_without_shapes_path}")