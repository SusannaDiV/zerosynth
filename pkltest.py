import pickle

def quick_peek_structure(pkl_path):
    with open(pkl_path, 'rb') as f:
        # Load just the first object to see its structure
        try:
            obj = pickle.load(f)
            print("\nTop level attributes:")
            for attr in dir(obj):
                if not attr.startswith('_'):  # Only show public attributes
                    print(f"- {attr}")
                    
            # Try to peek at molecules and fingerprints
            if hasattr(obj, 'molecules'):
                print(f"\nNumber of molecules: {len(obj.molecules)}")
                if len(obj.molecules) > 0:
                    print("First molecule attributes:", dir(obj.molecules[0]))
            
            if hasattr(obj, 'fingerprints'):
                print(f"\nFingerprint shape:", obj.fingerprints.shape if hasattr(obj.fingerprints, 'shape') else 'N/A')
                
        except Exception as e:
            print(f"Error reading pickle: {e}")

if __name__ == "__main__":
    quick_peek_structure("data/processed/all/fpindex_gaussian_ph4.pkl")