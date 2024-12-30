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
import psutil
import gc
import multiprocessing as mp
import glob
import re

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

def process_single_rotation(mol, rotation_mat, atom_stamp, cavity, protein, resolution=0.5, box_size=15):
    """Process a single rotation of a molecule with pre-computed conformers"""
    try:
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
        
        del copied_protein
        return {
            'mol': mol,
            'shape': centered_shape,
            'protein_grid': protein_grid.squeeze()
        }
        
    except Exception as e:
        print(f"Failed to process rotation: {str(e)}")
        return None

def process_molecule_batch(mol_batch, atom_stamp):
    """Process a batch of molecules with parallel rotations"""
    results = []
    for mol in mol_batch:
        if mol is None:
            continue
            
        try:
            
            # Generate 3D conformer once
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            
            # Create cavity and protein once
            cavity = Chem.Mol(mol)
            protein = Chem.Mol(mol)
            cavity_centroid = get_mol_centroid(cavity)
            cavity = centralize(cavity)
            
            # Process all rotations in parallel
            with mp.Pool(processes=min(24, mp.cpu_count())) as rotation_pool:
                rotation_results = rotation_pool.starmap(
                    process_single_rotation,
                    [(mol, rot_mat, atom_stamp, cavity, protein) 
                     for rot_mat in ROTATIONS]
                )
            
            # Collect valid results
            results.extend([r for r in rotation_results if r is not None])
            
            # Clean up
            del cavity, protein
            gc.collect()
            
        except Exception as e:
            print(f"Failed to process molecule: {str(e)}")
            continue
            
    return results

def main():
    input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
    old_output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation"  # Absolute path to old directory
    output_dir = "/itet-stor/sdivita/net_scratch/shitong/ChemProjector/data/shapes"  # Your new directory
    os.makedirs(output_dir, exist_ok=True)

    # Find existing intermediate files in BOTH directories with debug prints
    print(f"Checking old directory: {old_output_dir}")
    old_files = glob.glob(os.path.join(old_output_dir, 'dataset_batch_*.pkl'))
    print(f"Found {len(old_files)} files in old directory")
    for f in old_files:
        print(f"Old file: {f}")

    print(f"\nChecking new directory: {output_dir}")
    new_files = glob.glob(os.path.join(output_dir, 'dataset_batch_*.pkl'))
    print(f"Found {len(new_files)} files in new directory")
    for f in new_files:
        print(f"New file: {f}")

    existing_files = old_files + new_files
    
    last_processed = 0
    if existing_files:
        numbers = [int(re.search(r'dataset_batch_(\d+).pkl', f).group(1)) for f in existing_files]
        last_processed = max(numbers)
        print(f"\nFound total of {len(existing_files)} files. Resuming from {last_processed} processed molecules")
    
    # Stream molecules from SDF
    batch_size = 1000
    current_batch = []
    total_processed = last_processed  # Start from last processed count
    batch_files = existing_files  # Keep track of existing files
    processed_mols = 0
    current_mol_batch = []
    
    # Skip already processed molecules
    mol_supplier = Chem.ForwardSDMolSupplier(input_sdf, removeHs=False)
    for _ in range(last_processed):
        try:
            next(mol_supplier)
        except StopIteration:
            print("Reached end of file while skipping. All molecules already processed.")
            break
    
    # Process remaining molecules
    for mol in tqdm(mol_supplier):
        if mol is None:
            continue
            
        current_mol_batch.append(mol)
        
        # Process when we have enough molecules
        if len(current_mol_batch) >= 4:
            atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
            results = process_molecule_batch(current_mol_batch, atom_stamp)
            current_batch.extend(results)
            
            # Save batch when it reaches batch_size
            if len(current_batch) >= batch_size:
                total_processed += len(current_batch)
                batch_file = os.path.join(output_dir, f'dataset_batch_{total_processed}.pkl')
                print(f"Saving batch of {len(current_batch)} results (total: {total_processed})")
                
                with open(batch_file, 'wb') as fw:
                    pkl.dump(current_batch, fw)
                batch_files.append(batch_file)
                current_batch = []
                gc.collect()
            
            current_mol_batch = []
            processed_mols += 4
    
    # Process any remaining molecules
    if current_mol_batch:
        atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
        results = process_molecule_batch(current_mol_batch, atom_stamp)
        current_batch.extend(results)
    
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
    with open(os.path.join(output_dir, 'dataset2.pkl'), 'wb') as fw:
        pkl.dump(all_results, fw)
    print("Processing complete!")

if __name__ == "__main__":
    main() 