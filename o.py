from chemprojector.data.common import visualize_shape_generation
import os

# Create visualization directory
os.makedirs("shape_visualization", exist_ok=True)

# Test with a simple molecule
# Example usage
smiles = "Cc1ccc(N(C(c2ccc(Cc3nsc(NC(=O)C(C)(C)OC(C)C)n3)cc2)C(C)(C(=O)c2ccc(B(O)O)o2)c2ccccc2)S(C)(=O)=O)c(NS(=O)(=O)CC2CC3(CCC3)CCO2)c1"  # Aspirin
visualize_shape_generation(smiles, save_path="shape_visualizations")