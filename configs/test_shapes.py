import pickle
import torch
from tqdm import tqdm

def verify_shape_patches(fpindex_path: str):
    print(f"Loading fingerprint index from {fpindex_path}...")
    with open(fpindex_path, "rb") as f:
        fpindex = pickle.load(f)
    
    total_molecules = len(fpindex._shape_patches)
    zero_patches = 0
    non_zero_patches = 0
    zero_mol_indices = []
    
    print("\nAnalyzing shape patches...")
    for idx, patches in tqdm(fpindex._shape_patches.items()):
        if not isinstance(patches, torch.Tensor):
            print(f"Warning: Patches for molecule {idx} is not a tensor!")
            continue
            
        if torch.all(patches == 0):
            zero_patches += 1
            zero_mol_indices.append(idx)
        else:
            non_zero_patches += 1
            # Print some statistics for the first few non-zero patches
            if non_zero_patches <= 5:
                print(f"\nMolecule {idx} stats:")
                print(f"Shape: {patches.shape}")
                print(f"Min: {patches.min():.4f}")
                print(f"Max: {patches.max():.4f}")
                print(f"Mean: {patches.mean():.4f}")
                print(f"Non-zero elements: {torch.count_nonzero(patches).item()}")
    
    print("\nSummary:")
    print(f"Total molecules: {total_molecules}")
    print(f"Molecules with all zero patches: {zero_patches} ({zero_patches/total_molecules*100:.2f}%)")
    print(f"Molecules with non-zero patches: {non_zero_patches} ({non_zero_patches/total_molecules*100:.2f}%)")
    
    if len(zero_mol_indices) > 0:
        print("\nFirst 10 molecule indices with zero patches:", zero_mol_indices[:10])

if __name__ == "__main__":
    fpindex_path = "data/processed/all/fpindex_pharmacomit.pkl"
    verify_shape_patches(fpindex_path)