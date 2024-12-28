import os
import pickle as pkl
from shape_utils import get_atom_stamp, get_shape, ROTATIONS, centralize, get_mol_centroid, trans, get_binary_features
from tfbio_data import make_grid
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem
import copy
import numpy as np
from tqdm.auto import tqdm
from chemprojector.chem.mol import Molecule
from chemprojector.chem.featurize import atom_features_simple, bond_features_simple
        
def main():
    input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
    output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)

    supplier = Chem.SDMolSupplier(input_sdf)
    valid_mols = []
    for idx, mol in tqdm(enumerate(supplier), desc="Reading molecules"):
        if mol is not None:
            valid_mols.append(mol)
            if len(valid_mols) >= 10:  # Only take first 10 valid molecules
                break
    
    print(f"Processing {len(valid_mols)} molecules...")
    all_sample_shapes = []
    all_sample_n_o_f = []

    for idx, mol in tqdm(enumerate(valid_mols), desc="Processing molecules"):
        try:
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
            
            for i in range(24):  # 24 rotations
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

        except Exception as e:
            print(f"Failed to process molecule {idx}: {e}")
            continue

        # Save intermediate results every 1000 molecules
        if (idx + 1) % 8000 == 0:
            intermediate_data = []
            for mol_idx in range(len(all_sample_shapes)):
                mol = valid_mols[mol_idx]
                for rot_idx in range(len(all_sample_shapes[mol_idx])):
                    intermediate_data.append({
                        'mol': valid_mols[mol_idx],
                        'shape': all_sample_shapes[mol_idx][rot_idx],
                        'protein_grid': all_sample_n_o_f[mol_idx][rot_idx],
                    })
            print(f"Saving intermediate results after {idx + 1} molecules...")
            with open(os.path.join(output_dir, f'dataset_intermediate_{idx + 1}.pkl'), 'wb') as fw:
                pkl.dump(intermediate_data, fw)

    # Save final processed data
    processed_data = []
    for mol_idx in range(len(all_sample_shapes)):
        mol = valid_mols[mol_idx]
        for rot_idx in range(len(all_sample_shapes[mol_idx])):
            processed_data.append({
                'mol': valid_mols[mol_idx],
                'shape': all_sample_shapes[mol_idx][rot_idx],
                'protein_grid': all_sample_n_o_f[mol_idx][rot_idx],
            })

    print(f"Saving {len(processed_data)} shape pairs...")
    with open(os.path.join(output_dir, 'dataset2.pkl'), 'wb') as fw:
        pkl.dump(processed_data, fw)
    print("Processing complete!")

if __name__ == "__main__":
    main() 