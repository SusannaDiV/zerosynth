import pathlib
import pickle
import numpy as np
from collections import Counter

def analyze_fpindex(pkl_path):
    """Analyze the fingerprint index file"""
    print(f"Reading fingerprint index from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        fpindex = pickle.load(f)
    
    print("\nBasic Information:")
    print(f"Number of molecules: {len(fpindex.molecules)}")
    print(f"Fingerprint array shape: {fpindex._fp.shape}")
    
    # Analyze fingerprints
    fps = fpindex._fp
    
    print("\nFingerprint Statistics:")
    print(f"Mean value: {np.mean(fps):.4f}")
    print(f"Std deviation: {np.std(fps):.4f}")
    print(f"Min value: {np.min(fps)}")
    print(f"Max value: {np.max(fps)}")
    print(f"Number of non-zero elements: {np.count_nonzero(fps)}")
    print(f"Sparsity: {(1 - np.count_nonzero(fps)/fps.size)*100:.2f}%")
    
    # Check for zero vectors (failed fingerprints)
    zero_vectors = np.all(fps == 0, axis=1)
    num_zero_vectors = np.sum(zero_vectors)
    print(f"\nNumber of zero vectors (failed fingerprints): {num_zero_vectors}")
    print(f"Percentage of failed fingerprints: {(num_zero_vectors/len(fps))*100:.2f}%")
    
    # Look at some example fingerprints
    print("\nExample fingerprints:")
    for i in range(min(5, len(fps))):
        fp = fps[i]
        print(f"\nMolecule_{i}:")
        print(f"Sum: {np.sum(fp)}")
        print(f"Non-zero elements: {np.count_nonzero(fp)}")
        print(f"First 20 values: {fp[:20]}")
        
        # Print positions of non-zero elements in first 100 positions
        nonzero_pos = np.nonzero(fp[:100])[0]
        if len(nonzero_pos) > 0:
            print(f"First few non-zero positions: {nonzero_pos}")
            print(f"Values at these positions: {fp[nonzero_pos]}")
    
    # Distribution of values
    values = fps.flatten()
    value_counts = Counter(values)
    print("\nValue distribution:")
    for value, count in sorted(value_counts.most_common(10)):
        print(f"Value {value}: {count} times ({(count/values.size)*100:.2f}%)")
    
    # Additional analysis of fingerprint structure
    print("\nFingerprint structure analysis:")
    # Look at the distribution of values across different sections of the fingerprint
    section_size = 40  # Size of each feature pair section
    n_sections = fps.shape[1] // section_size
    for i in range(n_sections):
        section = fps[:, i*section_size:(i+1)*section_size]
        print(f"\nSection {i} (positions {i*section_size}-{(i+1)*section_size-1}):")
        print(f"Mean: {np.mean(section):.4f}")
        print(f"Non-zero elements: {np.count_nonzero(section)}")
        print(f"Max value: {np.max(section)}")

if __name__ == "__main__":
    pkl_path = pathlib.Path("data/processed/all/fpindex_ph4_ultimo.pkl")
    analyze_fpindex(pkl_path) 