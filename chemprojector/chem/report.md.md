# Understanding Pharmacophore Patch Shapes in ChemProjector

## Tensor Shape: `(7, 27 * 6)`

The pharmacophore patch tensor has a shape of `(7, 162)` where:

### First Dimension: 7
- Represents the number of rotations for each molecule
- Each molecule is processed in 7 different orientations
- This ensures rotational invariance in the representation

### Second Dimension: 27 * 6 = 162
Composed of two factors:

#### Grid Points: 27
- Each patch is a 3x3x3 cube
- Results in 27 spatial positions (3 * 3 * 3)
- Captures local spatial information around each point

#### Feature Types: 6
Each grid point can have 6 different types of pharmacophore features:
1. Hydrogen bond donor
2. Hydrogen bond acceptor
3. Positive charge
4. Negative charge
5. Aromatic
6. Hydrophobic

### Final Structure
- Shape: `(7, 162)`
- Data type: `float16` (for memory efficiency)
- Each rotation contains 27 spatial positions × 6 feature types = 162 values
- Features are flattened into a 1D array for each rotation

### Comparison with Shape Patches
- Shape patches have dimensions `(7, 27)`
- Only store one value per grid point (the shape value)
- Don't need the 6 feature types dimension