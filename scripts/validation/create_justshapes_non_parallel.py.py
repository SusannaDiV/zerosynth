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
import psutil
import gc
from allinacp4_ph4 import compute_fingerprint_from_mol

def get_shape_with_memory_check(cavity, atom_stamp, resolution, box_size):
    """Wrapper to check memory requirements before shape computation"""
    # Calculate required memory
    grid_points = int(2 * box_size / resolution) + 1
    required_memory = grid_points**3 * 4  # 4 bytes per float32
    
    # Check available memory (with 20% buffer)
    available_memory = psutil.virtual_memory().available * 0.8
    
    if required_memory > available_memory:
        raise MemoryError(f"Insufficient memory. Need {required_memory/1024:.0f} KiB, "
                         f"have {available_memory/1024:.0f} KiB available")
    
    return get_shape(cavity, atom_stamp, resolution, box_size)

def stream_process_molecule(mol, atom_stamp, resolution=0.5, box_size=15):
    """Process a single molecule and yield results immediately"""
    try:
        # Calculate fingerprint first
        fingerprint = compute_fingerprint_from_mol(mol).flatten()
        
        # Generate 3D conformer if needed
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        cavity = Chem.Mol(mol)
        protein = Chem.Mol(mol)
        
        cavity_centroid = get_mol_centroid(cavity)
        cavity = centralize(cavity)
        
        # Process rotations in smaller batches
        rotation_batch_size = 4
        for batch_idx in range(0, 24, rotation_batch_size):
            batch_rotations = ROTATIONS[batch_idx:batch_idx + rotation_batch_size]
            
            for rotation_mat in batch_rotations:
                copied_cavity = copy.deepcopy(cavity)
                copied_protein = copy.deepcopy(protein)
                
                cavity_conformer = copied_cavity.GetConformer()
                rotation = np.zeros((4, 4))
                rotation[:3, :3] = rotation_mat
                rdMolTransforms.TransformConformer(cavity_conformer, rotation)
                
                curr_cavity_shape = get_shape_with_memory_check(
                    copied_cavity, atom_stamp, resolution, box_size
                )
                
                del copied_cavity
                gc.collect()
                
                grid_size = 21
                start_idx = curr_cavity_shape.shape[0]//2 - grid_size//2
                end_idx = start_idx + grid_size
                centered_shape = curr_cavity_shape[
                    start_idx:end_idx,
                    start_idx:end_idx,
                    start_idx:end_idx
                ]
                
                protein_coords, protein_features = get_binary_features(copied_protein, -1, False)
                protein_grid = make_grid(protein_coords, protein_features, resolution, box_size)[0]
                
                yield {
                    'mol': mol,
                    'shape': centered_shape,
                    'protein_grid': protein_grid.squeeze(),
                    'fingerprint': fingerprint
                }
                
                del copied_protein
                gc.collect()
                
    except Exception as e:
        print(f"Failed to process molecule: {str(e)}")

def main():
    input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
    output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)

    # Stream molecules from SDF
    batch_size = 1000
    current_batch = []
    total_processed = 0
    batch_files = []  # Keep track of batch files
    processed_mols = 0  # Counter for processed molecules
    
    for mol in tqdm(Chem.ForwardSDMolSupplier(input_sdf, removeHs=False)):
        if mol is None:
            continue
            
        atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
        
        # Process each rotation and stream results
        for result in stream_process_molecule(mol, atom_stamp):
            current_batch.append(result)
            
            # Save batch when it reaches batch_size
            if len(current_batch) >= batch_size:
                total_processed += len(current_batch)
                batch_file = os.path.join(output_dir, f'dataset_batch_{total_processed}.pkl')
                print(f"Saving batch of {len(current_batch)} results (total: {total_processed})")
                
                with open(batch_file, 'wb') as fw:
                    pkl.dump(current_batch, fw)
                batch_files.append(batch_file)  # Track the batch file
                current_batch = []
                gc.collect()
        
        processed_mols += 1
        if processed_mols >= 100:  # Stop after 100 molecules
            break
    
    # Save any remaining results
    if current_batch:
        total_processed += len(current_batch)
        batch_file = os.path.join(output_dir, f'dataset_batch_{total_processed}.pkl')
        with open(batch_file, 'wb') as fw:
            pkl.dump(current_batch, fw)
        batch_files.append(batch_file)

    # Concatenate all batches into final dataset
    print("Combining batches into final dataset...")
    all_results = []
    for batch_file in tqdm(batch_files):
        with open(batch_file, 'rb') as fr:
            batch_data = pkl.load(fr)
            all_results.extend(batch_data)
        os.remove(batch_file)  # Clean up batch file after loading
    
    print(f"Saving final dataset with {len(all_results)} results...")
    with open(os.path.join(output_dir, 'dataset3.pkl'), 'wb') as fw:
        pkl.dump(all_results, fw)
    print("Processing complete!")

if __name__ == "__main__":
    main() 