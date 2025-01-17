import pickle
import torch
from tqdm import tqdm
from chemprojector.data.collate import collate_shape_patches

def test_collate_patches(fpindex_path: str):
    print(f"Loading fingerprint index from {fpindex_path}...")
    
    with open(fpindex_path, "rb") as f:
        fpindex = pickle.load(f)
    
    # Get a batch of patches
    batch_size = 32
    patches_list = []
    indices = []
    
    print("\nGathering batch of patches...")
    for idx, patches in tqdm(list(fpindex._shape_patches.items())[:batch_size]):
        patches = patches.cpu()
        patches_list.append(patches)
        indices.append(idx)
    
    print("\nPre-collation stats:")
    for i, (idx, patches) in enumerate(zip(indices, patches_list)):
        zero_percentage = (patches == 0).float().mean().item() * 100
        print(f"Molecule {idx}:")
        print(f"Shape: {patches.shape}")
        print(f"Zero percentage: {zero_percentage:.2f}%")
        print(f"Non-zero elements: {torch.count_nonzero(patches).item()}")
        if i >= 4:  # Only show first 5
            break
    
    print("\nCollating patches...")
    collated = collate_shape_patches(patches_list, max_size=None)
    
    print("\nPost-collation stats:")
    for i in range(min(5, len(indices))):
        patches = collated[i]
        zero_percentage = (patches == 0).float().mean().item() * 100
        print(f"Molecule {indices[i]}:")
        print(f"Shape: {patches.shape}")
        print(f"Zero percentage: {zero_percentage:.2f}%")
        print(f"Non-zero elements: {torch.count_nonzero(patches).item()}")
    
    print("\nOverall collated tensor:")
    print(f"Shape: {collated.shape}")
    print(f"Total zero percentage: {(collated == 0).float().mean().item() * 100:.2f}%")
    print(f"Memory usage: {collated.element_size() * collated.nelement() / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    fpindex_path = "data/processed/all/fpindex_pharmacomit.pkl"
    test_collate_patches(fpindex_path) 