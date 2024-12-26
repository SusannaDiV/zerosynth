import pathlib
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from chemprojector.chem.allinacp4_ph4 import compute_fingerprint_from_mol

def debug_first_molecule(sdf_path):
    """Debug conformer generation for the first molecule in SDF file"""
    print(f"Reading SDF file: {sdf_path}")
    
    # Read only first molecule
    mol_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(iter(mol_supplier))
    
    if mol is None:
        print("Error: Could not read first molecule")
        return
    
    # Print molecule info
    print("\nMolecule Information:")
    print(f"Name: {mol.GetProp('_Name') if mol.HasProp('_Name') else 'No name'}")
    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print(f"Number of bonds: {mol.GetNumBonds()}")
    print(f"Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    
    # Try to generate 3D conformer
    print("\nAttempting conformer generation...")
    mol_copy = Chem.Mol(mol)
    mol_copy = Chem.AddHs(mol_copy)
    
    try:
        # First try basic embedding
        print("Attempting basic embedding...")
        conf_id = AllChem.EmbedMolecule(mol_copy, randomSeed=42)
        print(f"Basic embedding result: {conf_id}")
        
        if conf_id == -1:
            # Try with different parameters
            print("\nAttempting embedding with more attempts...")
            conf_id = AllChem.EmbedMolecule(mol_copy, maxAttempts=1000, randomSeed=42)
            print(f"Extended embedding result: {conf_id}")
            
            if conf_id == -1:
                print("\nAttempting embedding with random coordinates...")
                conf_id = AllChem.EmbedMolecule(mol_copy, useRandomCoords=True, randomSeed=42)
                print(f"Random coords embedding result: {conf_id}")
        
        if conf_id >= 0:
            print("\nConformer generated successfully!")
            print("Attempting MMFF optimization...")
            try:
                AllChem.MMFFOptimizeMolecule(mol_copy)
                print("MMFF optimization successful!")
            except Exception as e:
                print(f"MMFF optimization failed: {str(e)}")
                print("Attempting UFF optimization...")
                try:
                    AllChem.UFFOptimizeMolecule(mol_copy)
                    print("UFF optimization successful!")
                except Exception as e:
                    print(f"UFF optimization failed: {str(e)}")
        
        # Try to compute fingerprint
        print("\nAttempting to compute fingerprint...")
        fingerprint = compute_fingerprint_from_mol(mol)
        print(f"Fingerprint shape: {fingerprint.shape}")
        print(f"Fingerprint sum: {np.sum(fingerprint)}")
        print(f"Non-zero elements: {np.count_nonzero(fingerprint)}")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")

if __name__ == "__main__":
    # Use the same path as in your create_fingerprint_index.py script
    sdf_path = pathlib.Path("/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf")
    debug_first_molecule(sdf_path) 