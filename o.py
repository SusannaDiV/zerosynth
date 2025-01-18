from chemprojector.data.common import visualize_shape_generation
import os

# Create visualization directory
os.makedirs("shape_visualization", exist_ok=True)

# Test with a simple molecule
# Example usage
smiles = "Cc1cscc1-c1nnnn1C[C@@H]1CC1(C(=O)C1C(N2CCCCC2CN(C)C2CCCCC2)CN(C(=O)OC(C)(C)C)C1(C)C(=O)O)c1nc(-c2cccc([N+](=O)[O-])c2N2CCC3(CCOC(C)C3)C2)n[nH]1"  # Aspirin
visualize_shape_generation(smiles, save_path="shape_visualizations")