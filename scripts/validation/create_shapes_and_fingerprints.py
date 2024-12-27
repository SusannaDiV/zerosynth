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

def get_graph_features(mol):
    """Extract graph features (atoms and bonds) from molecule"""
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),     # Number of bonds
            atom.GetFormalCharge(),  # Formal charge
            int(atom.GetChiralTag()),  # Chirality
            int(atom.GetIsAromatic()),  # Aromaticity
            atom.GetHybridization().real  # Hybridization state
        ]
        atom_features.append(features)
    
    # Bond features and connectivity
    bond_features = []
    edge_index = []  # [from_idx, to_idx] for each bond
    for bond in mol.GetBonds():
        features = [
            int(bond.GetBondType()),  # Bond type
            int(bond.GetIsConjugated()),  # Conjugation
            int(bond.IsInRing())  # Ring membership
        ]
        # Add both directions for each bond
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_features.extend([features, features])
    
    return {
        'atom_features': np.array(atom_features),
        'bond_features': np.array(bond_features) if bond_features else np.empty((0, 3)),
        'edge_index': np.array(edge_index).T if edge_index else np.empty((2, 0))
    }

def main():
    input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
    output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)

    supplier = Chem.SDMolSupplier(input_sdf)
    first_5_mols = []
    for idx, mol in enumerate(supplier):
        if mol is not None and len(first_5_mols) < 5:
            first_5_mols.append(mol)
    
    print(f"Processing {len(first_5_mols)} molecules...")
    all_sample_shapes = []
    all_sample_n_o_f = []
    all_fingerprints = []  # Added for fingerprints

    for idx, mol in enumerate(first_5_mols):
        try:
            # Get fingerprint first
            fingerprint = compute_fingerprint_from_mol(mol).flatten()
            
            # Generate 3D conformer if needed
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            is_2d = all(pos[2] == 0 for pos in positions)
            
            if is_2d:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)

            atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
            
            cavity = Chem.Mol(mol)
            protein = Chem.Mol(mol)
            
            cavity_centroid = get_mol_centroid(cavity)
            cavity = centralize(cavity)
            translation = trans(-cavity_centroid[0], -cavity_centroid[1], -cavity_centroid[2])
            protein_conformer = protein.GetConformer()
            rdMolTransforms.TransformConformer(protein_conformer, translation)

            sample_shapes = []
            sample_n_o_f = []
            
            for i in tqdm(range(24), desc=f"Processing rotations for molecule {idx}"):
                copied_cavity = copy.deepcopy(cavity)
                copied_protein = copy.deepcopy(protein)

                cavity_conformer = copied_cavity.GetConformer()
                protein_conformer = copied_protein.GetConformer()

                rotation_mat = ROTATIONS[i]
                rotation = np.zeros((4, 4))
                rotation[:3, :3] = rotation_mat
                rdMolTransforms.TransformConformer(cavity_conformer, rotation)
                rdMolTransforms.TransformConformer(protein_conformer, rotation)

                curr_cavity_shape = get_shape(copied_cavity, atom_stamp, 0.5, 15)
                grid_size = 21
                start_idx = curr_cavity_shape.shape[0]//2 - grid_size//2
                end_idx = start_idx + grid_size
                
                centered_shape = curr_cavity_shape[
                    start_idx:end_idx,
                    start_idx:end_idx,
                    start_idx:end_idx
                ]

                protein_coords, protein_features = get_binary_features(copied_protein, -1, False)
                protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 15)
                protein_grid = protein_grid.squeeze()

                sample_shapes.append(centered_shape)
                sample_n_o_f.append(protein_grid)

            all_sample_shapes.append(sample_shapes)
            all_sample_n_o_f.append(sample_n_o_f)
            all_fingerprints.append(fingerprint)

        except Exception as e:
            print(f"Failed to process molecule {idx}: {e}")
            continue

    # Save processed data
    processed_data = []
    for mol_idx in range(len(all_sample_shapes)):
        for rot_idx in range(len(all_sample_shapes[mol_idx])):
            processed_data.append({
                'mol': first_5_mols[mol_idx],  # Original molecule
                'shape': all_sample_shapes[mol_idx][rot_idx],  # Shape array (21,21,21)
                'protein_grid': all_sample_n_o_f[mol_idx][rot_idx],  # Protein grid array (21,21,21)
                'graph': get_graph_features(first_5_mols[mol_idx]),  # Graph structure
                'fingerprint': all_fingerprints[mol_idx]  # Target fingerprint (840,)
            })

    print(f"Saving {len(processed_data)} shape-fingerprint pairs...")
    with open(os.path.join(output_dir, 'shape_fingerprint_dataset1.pkl'), 'wb') as fw:
        pkl.dump(processed_data, fw)
    print("Processing complete!")

if __name__ == "__main__":
    main() 