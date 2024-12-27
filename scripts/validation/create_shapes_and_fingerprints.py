import os
import pickle as pkl
from shape_utils import get_atom_stamp, get_shape, ROTATIONS, centralize, get_mol_centroid, trans, get_binary_features
from tfbio_data import make_grid
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem
import copy
import numpy as np
from tqdm.auto import tqdm
from allinacp4_ph4 import compute_fingerprint_from_mol

def process_molecule_with_fingerprint(mol, atom_stamp):
    # Generate 3D conformer if needed
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    is_2d = all(pos[2] == 0 for pos in positions)
    
    if is_2d:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

    # Get fingerprint
    fingerprint = compute_fingerprint_from_mol(mol).flatten()
    
    # Get shape
    cavity = Chem.Mol(mol)
    cavity_centroid = get_mol_centroid(cavity)
    cavity = centralize(cavity)
    
    # Get shape with 24 rotations
    shapes = []
    for i in range(24):
        copied_cavity = copy.deepcopy(cavity)
        cavity_conformer = copied_cavity.GetConformer()
        
        rotation_mat = ROTATIONS[i]
        rotation = np.zeros((4, 4))
        rotation[:3, :3] = rotation_mat
        rdMolTransforms.TransformConformer(cavity_conformer, rotation)
        
        curr_shape = get_shape(copied_cavity, atom_stamp, 0.5, 15)
        grid_size = 21
        start_idx = curr_shape.shape[0]//2 - grid_size//2
        end_idx = start_idx + grid_size
        centered_shape = curr_shape[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        shapes.append(centered_shape)
    
    return shapes, fingerprint

def main():
    input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
    output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)

    supplier = Chem.SDMolSupplier(input_sdf)
    atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
    
    processed_data = []
    processed_count = 0
    
    for idx, mol in tqdm(enumerate(supplier), desc="Processing molecules"):
        if processed_count >= 5:  # Stop after 5 molecules
            break
            
        if mol is None:
            continue
            
        try:
            shapes, fingerprint = process_molecule_with_fingerprint(mol, atom_stamp)
            for shape in shapes:
                # Store data in the format expected by ShapePretrainingDataset
                processed_data.append({
                    'mol': shape,
                    'fingerprint': fingerprint
                })
            processed_count += 1
            print(f"Successfully processed molecule {idx}")
        except Exception as e:
            print(f"Failed to process molecule {idx}: {e}")
            continue
    
    print(f"Saving {len(processed_data)} shape-fingerprint pairs from {processed_count} molecules...")
    with open(os.path.join(output_dir, 'shape_fingerprint_dataset.pkl'), 'wb') as fw:
        pkl.dump(processed_data, fw)
    print("Processing complete!")

if __name__ == "__main__":
    main() 