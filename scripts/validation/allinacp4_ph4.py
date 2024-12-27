#!/usr/bin/env python3
#
# Copyright (C) 2022, Francois Berenger
# Tsuda laboratory, The University of Tokyo,
# 5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan.
#
# project molecules 3D conformers into the pharmacophore features/points space

import argparse, math, os, sys, time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ph4 feature SMARTS from the Pharmer software
# definitions from Lidio Meireles and David Ryan Koes
# Article: https://doi.org/10.1021/ci200097m
# Code: https://raw.githubusercontent.com/UnixJunkie/pharmer/master/pharmarec.cpp
aro_smarts = ["a1aaaaa1",
              "a1aaaa1"]

hbd_smarts = ["[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]",
              "[#8!H0&!$([OH][C,S,P]=O)]",
              "[#16!H0]"]

hba_smarts = ["[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]",
	      "[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]"]

pos_smarts = ["[+,+2,+3,+4]",
              "[$(CC)](=N)N", # amidine
	      "[$(C(N)(N)=N)]", # guanidine
              "[$(n1cc[nH]c1)]"]

neg_smarts = ["[-,-2,-3,-4]",
              "C(=O)[O-,OH,OX1]",
              "[$([S,P](=O)[O-,OH,OX1])]",
	      "c1[nH1]nnn1",
              "c1nn[nH1]n1",
              "C(=O)N[OH1,O-,OX1]",
              "C(=O)N[OH1,O-]",
	      "CO(=N[OH1,O-])",
              "[$(N-[SX4](=O)(=O)[CX4](F)(F)F)]"] # trifluoromethyl sulfonamide

hyd_smarts = ["a1aaaaa1",
	      "a1aaaa1",
	      # branched terminals as one point
	      "[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
	      "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
	      "*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
	      # simple rings only; need to combine points to get good results for 3d structures
	      "[C&r3]1~[C&r3]~[C&r3]1",
	      "[C&r4]1~[C&r4]~[C&r4]~[C&r4]1",
	      "[C&r5]1~[C&r5]~[C&r5]~[C&r5]~[C&r5]1",
	      "[C&r6]1~[C&r6]~[C&r6]~[C&r6]~[C&r6]~[C&r6]1",
	      "[C&r7]1~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]1",
	      "[C&r8]1~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]1",
	      # aliphatic chains
	      "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
	      "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
	      "[$([CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
	      # sulfur (apparently)
	      "[$([S]~[#6])&!$(S~[!#6])]"]

FEATURE_TYPES = ['ARO', 'HBD', 'HBA', 'POS', 'NEG', 'HYD']
GRID_SIZE = 40
GRID_RANGE = (-20, 40)

def get_grid_index(coord, min_val, max_val, n_bins):
    bin_size = (max_val - min_val) / n_bins
    idx = int((coord - min_val) / bin_size)
    return max(0, min(n_bins - 1, idx))

def pattern_of_smarts(s):
    return Chem.MolFromSmarts(s)

# compile all SMARTS
aro_patterns = list(map(pattern_of_smarts, aro_smarts))
hbd_patterns = list(map(pattern_of_smarts, hbd_smarts))
hba_patterns = list(map(pattern_of_smarts, hba_smarts))
pos_patterns = list(map(pattern_of_smarts, pos_smarts))
neg_patterns = list(map(pattern_of_smarts, neg_smarts))
hyd_patterns = list(map(pattern_of_smarts, hyd_smarts))

# geometric center of a matched pattern
# WARNING: single-conformer molecule is assumed
def average_match(mol, matched_pattern):
    avg_x = 0.0
    avg_y = 0.0
    avg_z = 0.0
    count = float(len(matched_pattern))
    conf0 = mol.GetConformer(0)
    for i in matched_pattern:
        xyz = conf0.GetAtomPosition(i)
        avg_x += xyz.x
        avg_y += xyz.y
        avg_z += xyz.z
    center = (avg_x / count,
              avg_y / count,
              avg_z / count)
    return center

def find_matches(mol, patterns):
    res = []
    for pat in patterns:
        # get all matches for that pattern
        matched = mol.GetSubstructMatches(pat)
        for m in matched:
            # get the center of each matched group
            avg = average_match(mol, m)
            res.append(avg)
    return res

def find_ARO(mol):
    return find_matches(mol, aro_patterns)

def find_HBD(mol):
    return find_matches(mol, hbd_patterns)

def find_HBA(mol):
    return find_matches(mol, hba_patterns)

def find_POS(mol):
    return find_matches(mol, pos_patterns)

def find_NEG(mol):
    return find_matches(mol, neg_patterns)

def find_HYD(cluster_HYD, mol):
    hydros = find_matches(mol, hyd_patterns)
    if not cluster_HYD:
        return hydros
    else:
        # regroup all hydrophobic features within 2.0A
        res = []
        n = len(hydros)
        idx2cluster = list(range(n))
        idx2cluster = list(range(n))
        for i in range(n):
            h_i = hydros[i]
            cluster_id = idx2cluster[i]
            for j in range(i+1, n):
                h_j = hydros[j]
                if euclid(h_i, h_j) <= 2.0:
                    # same cluster
                    idx2cluster[j] = cluster_id
        cluster_ids = set(idx2cluster)
        for cid in cluster_ids:
            group = []
            for i, h in enumerate(hydros):
                if idx2cluster[i] == cid:
                    group.append(h)
            res.append(average(group))
        return res

def euclid(xyz0, xyz1):
    x0, y0, z0 = xyz0
    x1, y1, z1 = xyz1
    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def average(l):
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    n = float(len(l))
    for (x, y, z) in l:
        sum_x += x
        sum_y += y
        sum_z += z
    return (sum_x / n,
            sum_y / n,
            sum_z / n)

def get_feature_points(mol, patterns):
    """Helper function to get feature points using patterns"""
    return find_matches(mol, patterns)

def get_hydrophobic_points(mol, patterns):
    """Helper function to get hydrophobic points with optional clustering"""
    return find_HYD(True, mol)  # Always use clustering by default

def compute_pairwise_distances(points1, points2):
    """Compute pairwise distances between two sets of points"""
    distances = []
    for p1 in points1:
        for p2 in points2:
            distances.append(euclid(p1, p2))
    return distances

def compute_fingerprint_from_mol(mol, cluster_HYD=True):
    """Compute pharmacophore fingerprint from molecule"""
    # Create a copy of the molecule to avoid modifying the original
    mol = Chem.Mol(mol)
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    try:
        # Try to generate conformer
        conf_success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if conf_success == -1:
            raise ValueError("Conformer generation failed")
            
        # Try MMFF optimization first
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            # Fallback to UFF if MMFF fails
            AllChem.UFFOptimizeMolecule(mol)
            
        # Get feature points
        aromatics = find_ARO(mol)
        donors = find_HBD(mol)
        acceptors = find_HBA(mol)
        positives = find_POS(mol)
        negatives = find_NEG(mol)
        hydrophobes = find_HYD(cluster_HYD, mol)
        
        # Compute pairwise distances and create fingerprint
        feature_points = [aromatics, donors, acceptors, positives, negatives, hydrophobes]
        n_features = len(FEATURE_TYPES)
        fingerprint = np.zeros((n_features * (n_features + 1) // 2) * GRID_SIZE, dtype=np.uint8)
        
        idx = 0
        for i in range(n_features):
            for j in range(i, n_features):
                points1 = feature_points[i]
                points2 = feature_points[j]
                if points1 and points2:
                    distances = compute_pairwise_distances(points1, points2)
                    hist, _ = np.histogram(distances, bins=GRID_SIZE, range=GRID_RANGE)
                    fingerprint[idx:idx + GRID_SIZE] = hist
                idx += GRID_SIZE
                
        # Verify fingerprint dimension
        expected_dim = (n_features * (n_features + 1) // 2) * GRID_SIZE
        if len(fingerprint) != expected_dim:
            print(f"Warning: Fingerprint dimension mismatch. Got {len(fingerprint)}, expected {expected_dim}")
            fingerprint = np.zeros(expected_dim, dtype=np.uint8)
            
        return fingerprint
        
    except Exception as e:
        print(f"Warning: Failed to compute pharmacophore fingerprint: {str(e)}")
        # Return zero vector in case of failure
        n_features = len(FEATURE_TYPES)
        return np.zeros((n_features * (n_features + 1) // 2) * GRID_SIZE, dtype=np.uint8)

def prfx_print(prfx, out, positions_3d):
    for (x, y, z) in positions_3d:
        out.write("%s %g %g %g\n" % (prfx, x, y, z))

def bild_print(out, color, trans, radius, feats):
    if len(feats) > 0:
        out.write(".color %s\n" % color)
        out.write(".transparency %g\n" % trans)
        for (x, y, z) in feats:
            out.write(".sphere %g %g %g %g\n" % (x, y, z, radius))

def bild_print_ARO(out, feats):
    bild_print(out, "green", 0.75, 1.5, feats)

def bild_print_HYD(out, feats):
    bild_print(out, "grey", 0.75, 1.5, feats)

def bild_print_HBD(out, feats):
    bild_print(out, "white", 0.75, 1.25, feats)

def bild_print_HBA(out, feats):
    bild_print(out, "orange", 0.75, 1.25, feats)

def bild_print_POS(out, feats):
    bild_print(out, "blue", 0.75, 1.0, feats)

def bild_print_NEG(out, feats):
    bild_print(out, "red", 0.75, 1.0, feats)

def print_ARO(out, aromatics):
    prfx_print("ARO", out, aromatics)

def print_HBD(out, donors):
    prfx_print("HBD", out, donors)

def print_HBA(out, acceptors):
    prfx_print("HBA", out, acceptors)

def print_POS(out, positives):
    prfx_print("POS", out, positives)

def print_NEG(out, negatives):
    prfx_print("NEG", out, negatives)

def print_HYD(out, hydrophobes):
    prfx_print("HYD", out, hydrophobes)

# better than default readline() never throwing an exception
def read_line_EOF(input):
    line = input.readline()
    if line == "":
        raise EOFError
    else:
        return line

# list all molecule names
def names_of_sdf_file(input_fn):
    try:
        with open(input_fn, 'r') as input:
            fst_name = read_line_EOF(input).strip()
            yield fst_name
            while True:
                line = read_line_EOF(input).strip()
                while line != "$$$$":
                    line = read_line_EOF(input).strip()
                next_name = read_line_EOF(input).strip()
                yield next_name
    except EOFError:
        pass

def path_prepend(dir, fn):
    if dir == '':
        return fn
    else:
        return (dir + '/' + fn)

def bild_output(input_dir, mol_name,
                aromatics, donors, acceptors,
                positives, negatives, hydrophobes):
    bild_fn = path_prepend(input_dir, mol_name + ".bild")
    with open(bild_fn, 'w') as bild_out:
        bild_print_ARO(bild_out, aromatics)
        bild_print_HBD(bild_out, donors)
        bild_print_HBA(bild_out, acceptors)
        bild_print_POS(bild_out, positives)
        bild_print_NEG(bild_out, negatives)
        bild_print_HYD(bild_out, hydrophobes)

if __name__ == '__main__':
    before = time.time()
    # CLI options parsing
    parser = argparse.ArgumentParser(
        description = "compute pharmacophore features for 3D molecules")
    parser.add_argument("-i", metavar = "input.sdf", dest = "input_fn",
                        help = "conformers input file")
    parser.add_argument("-o", metavar = "output.ph4", dest = "output_fn",
                        help = "ph4 features output file")
    parser.add_argument('--bild', dest='output_bild',
                        action='store_true', default=False,
                        help = "output BILD files for visu in chimera")
    parser.add_argument('--no-group', dest='cluster_HYD',
                        action='store_false', default=True,
                        help = "turn OFF grouping of HYD features")
    parser.add_argument('--permissive', dest='sanitize',
                        action='store_false', default=True,
                        help = "turn OFF rdkit valence check")
    # parse CLI ---------------------------------------------------------------
    if len(sys.argv) == 1:
        # user has no clue of what to do -> usage
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    input_fn = args.input_fn
    output_fn = args.output_fn
    mol_names = names_of_sdf_file(input_fn)
    input_dir = os.path.dirname(input_fn)
    sanitize = args.sanitize
    mol_supplier = Chem.SDMolSupplier(input_fn, sanitize=sanitize)
    output_bild = args.output_bild
    cluster_HYD = args.cluster_HYD
    # parse CLI end -----------------------------------------------------------
    count = 0
    errors = 0
    with open(output_fn, 'w') as out:
        for mol, name in zip(mol_supplier, mol_names):
            if not sanitize:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol,
                                 Chem.SANITIZE_SYMMRINGS | \
                                 Chem.SANITIZE_SETCONJUGATION | \
                                 Chem.SANITIZE_SETHYBRIDIZATION)
            # print("%d atoms" % mol.GetNumHeavyAtoms(), file=sys.stderr)
            if mol == None:
                errors += 1
            else:
                fingerprint = compute_fingerprint_from_mol(mol, cluster_HYD)
                out.write("%d:%s\n" % (len(fingerprint), name))
                print_ARO(out, find_ARO(mol))
                print_HBD(out, find_HBD(mol))
                print_HBA(out, find_HBA(mol))
                print_POS(out, find_POS(mol))
                print_NEG(out, find_NEG(mol))
                print_HYD(out, find_HYD(cluster_HYD, mol))
                if output_bild:
                    bild_output(input_dir, name,
                                find_ARO(mol), find_HBD(mol), find_HBA(mol),
                                find_POS(mol), find_NEG(mol), find_HYD(cluster_HYD, mol))
            count += 1
    after = time.time()
    dt = after - before
    #print("%d molecules @ %.2fHz; %d errors" % (count, count / dt, errors), file=sys.stderr)