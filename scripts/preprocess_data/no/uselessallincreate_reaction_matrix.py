#!/usr/bin/env python3

import pathlib
import os
import numpy as np
import joblib
from tqdm.auto import tqdm
import pickle
import tempfile
from rdkit import Chem
from collections.abc import Iterable
import signal
import sys

from chemprojector.chem.mol import Molecule, read_mol_file
from chemprojector.chem.reaction import Reaction, read_reaction_file

def stream_molecules(sdf_file):
    """Stream molecules from SDF file to reduce memory usage"""
    for mol in Chem.ForwardSDMolSupplier(str(sdf_file), removeHs=False):
        if mol is not None:
            yield Molecule(mol)
            del mol

def compute_matrix_streaming(molecule_stream, reactions, total_molecules, batch_size=1024, checkpoint_file=None):
    """Compute reaction matrix using streaming with checkpoint support"""
    global global_processed_count, global_matrix, global_checkpoint_file
    global_checkpoint_file = checkpoint_file
    
    start_idx = 0
    if checkpoint_file and checkpoint_file.exists():
        print(f"Found checkpoint file {checkpoint_file}")
        try:
            with np.load(checkpoint_file) as checkpoint:
                if 'processed_count' in checkpoint:
                    start_idx = int(checkpoint['processed_count'])
                    if start_idx >= total_molecules:
                        print("All molecules already processed, loading from checkpoint...")
                        matrix = checkpoint['matrix']
                        print("Checkpoint loaded successfully")
                        return matrix
                    print(f"Resuming from molecule {start_idx}")
                    global_processed_count = start_idx
                    
                    print(f"Skipping {start_idx} molecules...")
                    for _ in tqdm(range(start_idx), desc="Skipping molecules"):
                        next(molecule_stream)
                    print("Done skipping")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            start_idx = 0

    with tempfile.TemporaryDirectory() as tempdir_s:
        temp_fname = pathlib.Path(tempdir_s) / "matrix"
        matrix = np.memmap(
            str(temp_fname),
            dtype=np.uint8,
            mode="w+",
            shape=(total_molecules, len(reactions)),
        )
        global_matrix = matrix
        
        array_idx = start_idx
        
        try:
            for mol_idx, mol in tqdm(enumerate(molecule_stream, start=start_idx), 
                               total=total_molecules, 
                               initial=start_idx,
                               desc="Processing molecules"):
                try:
                    for j, reaction in enumerate(reactions):
                        flag = 0
                        for t in reaction.match_reactant_templates(mol):
                            flag |= 1 << t
                        matrix[array_idx, j] = flag
                    array_idx += 1
                    global_processed_count = array_idx
                    
                    if array_idx % batch_size == 0:
                        save_checkpoint(matrix, array_idx, checkpoint_file)
                        
                except Exception as e:
                    print(f"Failed to process molecule {mol_idx}: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving checkpoint...")
            if checkpoint_file:
                try:
                    save_checkpoint(matrix, array_idx, checkpoint_file)
                    print(f"Saved checkpoint at molecule {array_idx}")
                except Exception as e:
                    print(f"Failed to save checkpoint: {str(e)}")
            raise
            
        finally:
            if checkpoint_file:
                try:
                    save_checkpoint(matrix, array_idx, checkpoint_file)
                    print(f"Saved final checkpoint at molecule {array_idx}")
                except Exception as e:
                    print(f"Failed to save final checkpoint: {str(e)}")
        
        return np.array(matrix)

def save_checkpoint(matrix, count, checkpoint_file):
    temp_file = checkpoint_file.with_suffix('.tmp')
    np.savez(temp_file,
            matrix=matrix[:count],
            processed_count=count)
    temp_file.rename(checkpoint_file)

def signal_handler(signum, frame):
    if global_matrix is not None and global_checkpoint_file is not None:
        save_checkpoint(global_matrix, global_processed_count, global_checkpoint_file)
    sys.exit(1)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactant", type=str, required=True)
    parser.add_argument("--reaction", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/all/matrix_ph4.pkl")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoints/matrix_checkpoint.npz")
    args = parser.parse_args()
    
    output_path = pathlib.Path(args.output)
    checkpoint_path = pathlib.Path(args.checkpoint)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading reactions...")
    reactions = list(read_reaction_file(args.reaction))
    
    print("Counting molecules...")
    total_molecules = 0
    for mol in tqdm(Chem.ForwardSDMolSupplier(args.reactant)):
        if mol is not None:
            total_molecules += 1
    
    print(f"Found {total_molecules} molecules")
    print(f"Found {len(reactions)} reactions")
    
    print("Creating reaction matrix...")
    molecule_stream = stream_molecules(args.reactant)
    matrix = compute_matrix_streaming(
        molecule_stream,
        reactions,
        total_molecules,
        checkpoint_file=checkpoint_path
    )
    
    print("Creating ReactantReactionMatrix object...")
    from chemprojector.chem.matrix import ReactantReactionMatrix
    m = ReactantReactionMatrix(
        reactants=list(stream_molecules(args.reactant)),
        reactions=reactions,
        matrix=matrix
    )
    
    print("Saving matrix...")
    with open(output_path, "wb") as f:
        pickle.dump(m, f)
    
    print(f"Processed {total_molecules} molecules")
    print(f"Number of reactions: {len(reactions)}")
    print(f"Saved matrix to {output_path}")