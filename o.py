from chemprojector.data.common import visualize_shape_generation
import os

# Create visualization directory
os.makedirs("shape_visualization", exist_ok=True)

# Test with a simple molecule
# Example usage
smiles = "COC(=O)c1nnn(C(Cc2nnn(Cc3ccsc3)n2)c2ccccc2NC2CCN(c3ccc(Cl)cc3)C2)c1-c1cccc(-c2ccccc2C(=O)Nc2cccc(C(=O)Nc3ccc(C)cc3)c2)c1"  # Aspirin
visualize_shape_generation(smiles, save_path="shape_visualizations")