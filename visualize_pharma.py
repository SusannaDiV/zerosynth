import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem
from chemprojector.chem.mol import Molecule
from chemprojector.chem.fpindex import FingerprintIndex, FingerprintOption
import os

def visualize_molecule_features(sdf_path):
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Read first molecule from SDF
    print(f"Loading molecule from {sdf_path}")
    suppl = Chem.SDMolSupplier(sdf_path)
    rdmol = next(suppl)
    
    if rdmol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")
    
    # Convert to our Molecule class
    try:
        mol = Molecule(rdmol)
    except Exception as e:
        print(f"Error initializing Molecule: {e}")
        raise
        
    # Generate 3D conformation if not present
    if not mol._rdmol.GetNumConformers():
        print("Generating 3D conformation...")
        mol._rdmol = Chem.AddHs(mol._rdmol)
        AllChem.EmbedMolecule(mol._rdmol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol._rdmol)
        mol._rdmol = Chem.RemoveHs(mol._rdmol)
    
    # Get pharmacophore features
    try:
        ph4_features = mol.get_pharmacophore_features()
        print("Found pharmacophore features:")
        for feat_type, features in ph4_features.items():
            print(f"  {feat_type}: {len(features)}")
    except Exception as e:
        print(f"Error getting pharmacophore features: {e}")
        raise
    
    # Get shape patches
    fp_option = FingerprintOption(
        grid_size=0.5,
        box_size=15,
        patch_size=3
    )
    
    fpindex = FingerprintIndex([mol], fp_option)
    shape_patches = fpindex.shape_patches[id(mol)][0]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    
    # 1. Plot molecule with pharmacophore features
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot molecule atoms
    conf = mol._rdmol.GetConformer()
    positions = conf.GetPositions()
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                c='gray', s=50, alpha=0.5, label='Atoms')
    
    # Plot bonds
    for bond in mol._rdmol.GetBonds():
        id1 = bond.GetBeginAtomIdx()
        id2 = bond.GetEndAtomIdx()
        pos1 = positions[id1]
        pos2 = positions[id2]
        ax1.plot([pos1[0], pos2[0]], 
                 [pos1[1], pos2[1]], 
                 [pos1[2], pos2[2]], 
                 'k-', alpha=0.3)
    
    # Plot pharmacophore features
    colors = {
        'HBD': 'blue',
        'HBA': 'red',
        'ARO': 'purple',
        'HYD': 'yellow',
        'POS': 'cyan',
        'NEG': 'orange'
    }
    
    for feat_type, features in ph4_features.items():
        if features:
            positions = np.array([f['position'] for f in features])
            ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c=colors[feat_type], s=100, label=feat_type)
    
    ax1.set_title('Molecule with Pharmacophore Features')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Plot shape patches
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Convert patches to 3D grid points
    patch_size = fp_option.patch_size
    grid_size = int(2 * fp_option.max_dist / fp_option.grid_resolution) + 1
    
    # Create grid coordinates
    x = np.arange(grid_size) * fp_option.grid_resolution - fp_option.max_dist
    y = np.arange(grid_size) * fp_option.grid_resolution - fp_option.max_dist
    z = np.arange(grid_size) * fp_option.grid_resolution - fp_option.max_dist
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Plot patches as points with intensity-based color
    patch_values = shape_patches.reshape(-1)
    significant_points = patch_values > np.percentile(patch_values, 90)  # Show top 10% intensity
    
    scatter = ax2.scatter(X.flatten()[significant_points],
                         Y.flatten()[significant_points],
                         Z.flatten()[significant_points],
                         c=patch_values[significant_points],
                         cmap='viridis',
                         alpha=0.6)
    plt.colorbar(scatter, ax=ax2, label='Patch Intensity')
    
    ax2.set_title('Shape Patches')
    
    # Set equal aspects and labels
    for ax in [ax1, ax2]:
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig('data/nowmolecule_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mol, ph4_features, shape_patches

# Usage
if __name__ == "__main__":
    sdf_path = "data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
    mol, ph4_features, shape_patches = visualize_molecule_features(sdf_path)
    
    # Print feature statistics
    print("\nPharmacophore Feature Statistics:")
    for feat_type, features in ph4_features.items():
        print(f"{feat_type}: {len(features)} features")
    
    print(f"\nShape Patches: {len(shape_patches)} patches")
    print(f"Patch dimensions: {shape_patches.shape}")