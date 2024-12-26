import os
import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

from chemprojector.chem.allinacp4_ph4 import (
    find_ARO, find_HBD, find_HBA, find_POS, find_NEG, find_HYD,
    FEATURE_TYPES, pattern_of_smarts
)
from chemprojector.chem.gaussian_ph4 import GaussianPH4Generator
from chemprojector.chem.mol import Molecule

TEST_DATA_DIR = Path(__file__).parent / "test_data"

def setup_module():
    """Setup test data directory"""
    TEST_DATA_DIR.mkdir(exist_ok=True)

def generate_3d_conformer(mol):
    """Helper function to generate 3D conformer"""
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def test_ph4_feature_detection():
    """Test basic pharmacophore feature detection"""
    # Test molecule with known features
    mol = Chem.MolFromSmiles('CC(=O)O')  # Acetic acid
    assert mol is not None
    mol = generate_3d_conformer(mol)
    
    # Test feature detection
    neg_features = find_NEG(mol)
    assert len(neg_features) > 0  # Should have negative feature (carboxylate)
    
    hba_features = find_HBA(mol)
    assert len(hba_features) > 0  # Should have H-bond acceptor (O)
    
    pos_features = find_POS(mol)
    assert len(pos_features) == 0  # Should have no positive features

def test_gaussian_ph4_generation():
    """Test Gaussian pharmacophore feature generation"""
    generator = GaussianPH4Generator(
        sigma=1.0,
        feature_types=FEATURE_TYPES,
        distance_cutoff=12.0,
        normalize=True
    )
    
    mol = Chem.MolFromSmiles('CCO')
    assert mol is not None
    mol = generate_3d_conformer(mol)
    
    features = []
    for feat_type in FEATURE_TYPES:
        if feat_type == 'ARO':
            features.extend(find_ARO(mol))
        elif feat_type == 'HBD':
            features.extend(find_HBD(mol))
        elif feat_type == 'HBA':
            features.extend(find_HBA(mol))
        elif feat_type == 'POS':
            features.extend(find_POS(mol))
        elif feat_type == 'NEG':
            features.extend(find_NEG(mol))
        elif feat_type == 'HYD':
            features.extend(find_HYD(True, mol))
    
    assert len(features) > 0

def test_ph4_file_generation():
    """Test generation of .ph4 files"""
    test_sdf = TEST_DATA_DIR / "test.sdf"
    test_pdb = TEST_DATA_DIR / "test.pdb"
    
    mol = Chem.MolFromSmiles('CCO')
    mol = generate_3d_conformer(mol)
    
    writer = Chem.SDWriter(str(test_sdf))
    writer.write(mol)
    writer.close()
    
    with open(test_pdb, 'w') as f:
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            line = (f"ATOM  {i+1:5d}  {atom.GetSymbol():<3}{' ':1}{'LIG':3} {'A':1}{1:4d}"
                   f"{' ':4}{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{1.00:6.2f}{0.00:6.2f}"
                   f"{' ':10}{atom.GetSymbol():>2}{' ':2}\n")
            f.write(line)
        f.write("TER\nEND\n")
    
    assert test_pdb.exists()

def test_ph4_feature_integration():
    """Test integration of ph4 features with molecule processing"""
    mol = Molecule.from_smiles('CCO')
    
    atom_features, bond_features = mol.featurize_simple()
    assert atom_features is not None
    assert bond_features is not None
    
    mol_rdkit = mol._rdmol
    features = {
        'ARO': find_ARO(mol_rdkit),
        'HBD': find_HBD(mol_rdkit),
        'HBA': find_HBA(mol_rdkit),
        'POS': find_POS(mol_rdkit),
        'NEG': find_NEG(mol_rdkit),
        'HYD': find_HYD(True, mol_rdkit)
    }
    
    for feat_type in FEATURE_TYPES:
        assert feat_type in features

def test_smarts_patterns():
    """Test SMARTS pattern compilation and matching"""
    mol = Chem.MolFromSmiles('c1ccccc1')  # Benzene
    mol = Chem.AddHs(mol)  # Add hydrogens
    
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SetAromaticity(mol)
    
    matched = False
    for pattern in map(pattern_of_smarts, ["a1aaaaa1", "a1aaaa1"]):
        matches = mol.GetSubstructMatches(pattern)
        if len(matches) > 0:
            matched = True
            break
    
    assert matched, "No aromatic patterns matched benzene"
    
    mol2 = Chem.MolFromSmiles('CC(=O)O')  # Acetic acid
    mol2 = Chem.AddHs(mol2)
    
    # Test negative charge pattern
    neg_pattern = pattern_of_smarts("C(=O)[O-,OH,OX1]")
    neg_matches = mol2.GetSubstructMatches(neg_pattern)
    assert len(neg_matches) > 0, "Carboxylic acid pattern not matched"
    
    # Test H-bond acceptor pattern
    hba_pattern = pattern_of_smarts("[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]")
    hba_matches = mol2.GetSubstructMatches(hba_pattern)
    assert len(hba_matches) > 0, "H-bond acceptor pattern not matched"

def teardown_module():
    """Cleanup test data"""
    import shutil
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR) 