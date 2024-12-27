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
from chemprojector.chem.mol import Molecule, FingerprintOption
from chemprojector.chem.featurize import atom_features_simple, bond_features_simple
from chemprojector.models.transformer.graph_transformer import GraphTransformer

def get_graph_features(mol):
    """Extract graph features using ChemProjector's featurization"""
    # Add Hydrogens for better graph representation
    mol = Chem.AddHs(mol)
    
    # Get atom features and positions
    atom_features = []
    positions = []
    conf = mol.GetConformer()
    
    for atom in mol.GetAtoms():
        features = [
            atom_features_simple(atom),
            atom.GetDegree(),     
            atom.GetFormalCharge(),
            int(atom.GetChiralTag()),
            int(atom.GetIsAromatic()),
            atom.GetHybridization().real,
            atom.GetNumImplicitHs(),
            int(atom.IsInRing()),
            atom.GetTotalValence()
        ]
        atom_features.append(features)
        
        # Get 3D positions
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
    
    # Bond features and connectivity
    bond_features = []
    edge_index = []
    for bond in mol.GetBonds():
        features = [
            bond_features_simple(bond),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo()),
            int(bond.GetBondDir())
        ]
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_features.extend([features, features])
    
    # Create attention mask (all True since single molecule)
    num_atoms = len(atom_features)
    attention_mask = np.ones((num_atoms, num_atoms), dtype=bool)
    
    # Remove Hydrogens after processing
    mol = Chem.RemoveHs(mol)
    
    return {
        'atom_features': np.array(atom_features),
        'bond_features': np.array(bond_features) if bond_features else np.empty((0, 5)),
        'edge_index': np.array(edge_index).T if edge_index else np.empty((2, 0)),
        'positions': np.array(positions),
        'attention_mask': attention_mask
    }
        
def main():
    input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
    output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)

    supplier = Chem.SDMolSupplier(input_sdf)
    first_5_mols = []
    for idx, mol in enumerate(supplier):
        if mol is not None and len(first_5_mols) < 20:
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
        mol = first_5_mols[mol_idx]
        graph_features = get_graph_features(mol)  # Get graph features once per molecule
        
        for rot_idx in range(len(all_sample_shapes[mol_idx])):
            processed_data.append({
                'mol': first_5_mols[mol_idx],
                'shape': all_sample_shapes[mol_idx][rot_idx],
                'protein_grid': all_sample_n_o_f[mol_idx][rot_idx],
                'graph_features': graph_features,  # Add graph features
                'fingerprint': all_fingerprints[mol_idx]
            })

    print(f"Saving {len(processed_data)} shape-fingerprint pairs...")
    with open(os.path.join(output_dir, 'shape_fingerprint_dataset1.pkl'), 'wb') as fw:
        pkl.dump(processed_data, fw)
    print("Processing complete!")

if __name__ == "__main__":
    main() 