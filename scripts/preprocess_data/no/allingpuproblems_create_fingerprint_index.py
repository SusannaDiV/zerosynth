import pathlib
import os
import numpy as np
import joblib
from tqdm.auto import tqdm
import pickle
import tempfile
from rdkit import Chem
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from allinacp4_ph4 import compute_fingerprint_from_mol
import signal
import sys

@dataclass(frozen=True)
class FingerprintOption:
    type: str = "ph4"
    dim: int = 840  # (6 * 7/2) feature pairs * (20/0.5) distance bins

class Molecule:
    def __init__(self, rdmol):
        self._rdmol = rdmol
        self.csmiles = Chem.MolToSmiles(rdmol, canonical=True)
        import hashlib
        self.csmiles_md5 = hashlib.md5(self.csmiles.encode()).digest()
    
    @cache
    def get_fingerprint(self, option):
        return compute_fingerprint_from_mol(self._rdmol).flatten()

def stream_molecules(sdf_file):
    """Stream molecules from SDF file to reduce memory usage"""
    for mol in Chem.ForwardSDMolSupplier(str(sdf_file), removeHs=False):
        if mol is not None:
            yield Molecule(mol)
            del mol

def compute_fingerprints_streaming(molecule_stream, total_molecules, fp_option, batch_size=16, checkpoint_file=None):
    """Compute fingerprints using streaming with checkpoint support"""
    global global_processed_count, global_fp, global_checkpoint_file
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
                        fingerprints = np.zeros((total_molecules, fp_option.dim), dtype=np.uint8)
                        chunk_size = 10000  
                        for i in tqdm(range(0, start_idx, chunk_size), desc="Loading checkpoint"):
                            end_idx = min(i + chunk_size, start_idx)
                            fingerprints[i:end_idx] = checkpoint['fingerprints'][i:end_idx]
                        print("Checkpoint loaded successfully")
                        return fingerprints
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
        temp_fname = pathlib.Path(tempdir_s) / "fingerprint"
        
        fp = np.memmap(
            str(temp_fname),
            dtype=np.uint8,
            mode="w+",
            shape=(total_molecules, fp_option.dim),
        )
        global_fp = fp
        
        array_idx = start_idx 
        
        try:
            for mol_idx, mol in tqdm(enumerate(molecule_stream, start=start_idx), 
                               total=total_molecules, 
                               initial=start_idx,
                               desc="Processing molecules"):
                try:
                    # Process single molecule
                    fingerprint = mol.get_fingerprint(fp_option).astype(np.uint8)
                    fp[array_idx] = fingerprint
                    array_idx += 1
                    global_processed_count = array_idx
                    
                    if checkpoint_file and (array_idx % 10000 == 0):
                        np.savez(checkpoint_file,
                                fingerprints=fp[:array_idx],
                                processed_count=array_idx)
                        print(f"Saved checkpoint at molecule {array_idx}")
                    
                    del fingerprint
                    del mol
                    
                except Exception as e:
                    print(f"Failed to process molecule at index {mol_idx}: {str(e)}")
                    continue
                
                if array_idx % 10 == 0:
                    import gc
                    gc.collect()
                    fp.flush()
                
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            
        finally:
            # Always try to save a final checkpoint
            if checkpoint_file and array_idx > start_idx:
                try:
                    np.savez(checkpoint_file,
                            fingerprints=fp[:array_idx],
                            processed_count=array_idx)
                    print(f"Saved final checkpoint at molecule {array_idx}")
                except Exception as e:
                    print(f"Failed to save final checkpoint: {str(e)}")
        
        return np.array(fp)

class FingerprintIndex:
    def __init__(self, sdf_file, fp_option, total_molecules, checkpoint_file=None):
        self._fp_option = fp_option
        molecule_stream = stream_molecules(sdf_file)
        
        print("Loading/Computing fingerprints...")
        self._fp = compute_fingerprints_streaming(
            molecule_stream, 
            total_molecules, 
            self._fp_option, 
            checkpoint_file=checkpoint_file
        )
        
        print("Constructing BallTree for similarity search...")
        print(f"Input shape: {self._fp.shape}, Memory usage: {self._fp.nbytes / 1e9:.2f} GB")
        
        from sklearn.neighbors import BallTree
        import psutil
        process = psutil.Process()
        
        print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")
        print("Building BallTree (this may take a while)...")
        
        with joblib.parallel_backend('threading', n_jobs=1):  
            with tqdm(total=1, desc="Building index") as pbar:
                self._tree = BallTree(self._fp, metric="manhattan")
                pbar.update(1)

def save_checkpoint(fp, count, checkpoint_file):
    temp_file = checkpoint_file.with_suffix('.tmp')
    np.savez(temp_file,
            fingerprints=fp[:count],
            processed_count=count)
    temp_file.rename(checkpoint_file)

def signal_handler(signum, frame):
    if global_fp is not None and global_checkpoint_file is not None:
        save_checkpoint(global_fp, global_processed_count, global_checkpoint_file)
    sys.exit(1)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-file", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/all/fpindex_ph4.pkl")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoints/fingerprint_checkpoint.npz")
    args = parser.parse_args()
    
    output_path = pathlib.Path(args.output)
    checkpoint_path = pathlib.Path(args.checkpoint)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    fp_option = FingerprintOption()
    
    print("Counting molecules...")
    total_molecules = 0
    for mol in tqdm(Chem.ForwardSDMolSupplier(args.sdf_file)):
        if mol is not None:
            total_molecules += 1
    
    print(f"Found {total_molecules} molecules")
    
    print("Creating fingerprint index...")
    fpindex = FingerprintIndex(
        args.sdf_file, 
        fp_option, 
        total_molecules, 
        checkpoint_file=checkpoint_path
    )
    
    print("Saving index...")
    total_size = os.path.getsize(args.sdf_file)  
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Saving") as pbar:
        with open(output_path, "wb") as f:
            pickle.dump(fpindex, f)
            pbar.update(total_size)
    
    print(f"Processed {total_molecules} molecules")
    print(f"Saved index to {output_path}")