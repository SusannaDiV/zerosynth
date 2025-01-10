import pickle
import numpy as np

# Load the dataset
with open('/itet-stor/sdivita/net_scratch/shitong/ChemProjector/data/processed/all/fpindex_gaussian_ph4.pkl', 'rb') as f:
    data = pickle.load(f)

# Function to print structure recursively
def print_structure(obj, level=0, max_items=3):
    indent = "  " * level
    
    if isinstance(obj, dict):
        print(f"{indent}Dict with {len(obj)} keys:")
        for k, v in obj.items():
            print(f"{indent}  {k}:")
            print_structure(v, level + 2)
    
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{type(obj).__name__} with {len(obj)} items:")
        if len(obj) > 0:
            print(f"{indent}First item:")
            print_structure(obj[0], level + 1)
            if len(obj) > max_items:
                print(f"{indent}... ({len(obj)-1} more items)")
    
    elif isinstance(obj, np.ndarray):
        print(f"{indent}numpy.ndarray with shape {obj.shape} and dtype {obj.dtype}")
    
    else:
        print(f"{indent}{type(obj).__name__}: {str(obj)[:100]}")

# Print the structure
print("\nDataset structure:")
print_structure(data)

# Print some statistics
print("\nDataset statistics:")
if isinstance(data, list):
    print(f"Total number of items: {len(data)}")
    if len(data) > 0 and isinstance(data[0], dict):
        print("\nKeys in first item:")
        for k, v in data[0].items():
            if isinstance(v, np.ndarray):
                print(f"- {k}: numpy array with shape {v.shape}")
            else:
                print(f"- {k}: {type(v)}")