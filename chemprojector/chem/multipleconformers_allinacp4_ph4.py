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
GRID_SIZE = 20
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
def average_match(mol, matched_pattern, conf_id=0):
    avg_x = 0.0
    avg_y = 0.0
    avg_z = 0.0
    count = float(len(matched_pattern))
    conf = mol.GetConformer(conf_id)
    for i in matched_pattern:
        xyz = conf.GetAtomPosition(i)
        avg_x += xyz.x
        avg_y += xyz.y
        avg_z += xyz.z
    center = (avg_x / count,
             avg_y / count,
             avg_z / count)
    return center

def find_matches(mol, patterns, conf_id=0):
    res = []
    for pat in patterns:
        matched = mol.GetSubstructMatches(pat)
        for m in matched:
            avg = average_match(mol, m, conf_id)
            res.append(avg)
    return res

def find_ARO(mol, conf_id=0):
    return find_matches(mol, aro_patterns, conf_id)

def find_HBD(mol, conf_id=0):
    return find_matches(mol, hbd_patterns, conf_id)

def find_HBA(mol, conf_id=0):
    return find_matches(mol, hba_patterns, conf_id)

def find_POS(mol, conf_id=0):
    return find_matches(mol, pos_patterns, conf_id)

def find_NEG(mol, conf_id=0):
    return find_matches(mol, neg_patterns, conf_id)

def find_HYD(cluster_HYD, mol, conf_id=0):
    hydros = find_matches(mol, hyd_patterns, conf_id)
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

def compute_fingerprint_from_mol(mol, n_confs=10):
    # Generate multiple conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # For reproducibility
    params.useSmallRingTorsions = True  # Better ring conformers
    params.useBasicKnowledge = True  # Use basic chemistry knowledge
    
    # Generate conformers
    cids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=n_confs,
        params=params,
        clearConfs=True,
        numThreads=0  # Use all available cores
    )
    
    if not cids:
        raise ValueError("Conformer generation failed")
        
    # Optimize all conformers
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    
    # Constants
    MAX_DIST = 20.0  # Maximum distance to consider
    BIN_SIZE = 0.5   # Distance discretization
    NUM_BINS = int(MAX_DIST / BIN_SIZE)
    n_feat_types = len(FEATURE_TYPES)
    n_channels = (n_feat_types * (n_feat_types + 1)) // 2
    
    # Store fingerprints for all conformers
    all_fps = []
    
    # Process each conformer
    for conf_id in range(mol.GetNumConformers()):
        features = []
        # Get features for this conformer
        for feat_type, finder in [
            ('ARO', find_ARO), ('HBD', find_HBD), 
            ('HBA', find_HBA), ('POS', find_POS),
            ('NEG', find_NEG), ('HYD', lambda m, cid: find_HYD(True, m, cid))
        ]:
            for coords in finder(mol, conf_id):
                features.append((feat_type, coords))
        
        # Initialize fingerprint for this conformer
        fp = np.zeros(n_channels * NUM_BINS, dtype=np.float32)
        
        # Calculate pairwise features
        for i, (type1, coords1) in enumerate(features):
            idx1 = FEATURE_TYPES.index(type1)
            for type2, coords2 in features[i+1:]:
                idx2 = FEATURE_TYPES.index(type2)
                
                # Get channel index (upper triangular)
                if idx1 <= idx2:
                    channel = (idx1 * n_feat_types - (idx1 * (idx1 - 1))//2) + (idx2 - idx1)
                else:
                    channel = (idx2 * n_feat_types - (idx2 * (idx2 - 1))//2) + (idx1 - idx2)
                
                # Distance binning with interpolation
                dist = np.linalg.norm(np.array(coords1) - np.array(coords2))
                if dist <= MAX_DIST:
                    bin_idx = dist / BIN_SIZE
                    bin_low = int(bin_idx)
                    bin_high = bin_low + 1
                    frac_high = bin_idx - bin_low
                    frac_low = 1.0 - frac_high
                    
                    if bin_low < NUM_BINS:
                        fp[channel * NUM_BINS + bin_low] += frac_low
                    if bin_high < NUM_BINS:
                        fp[channel * NUM_BINS + bin_high] += frac_high
        
        all_fps.append(fp)
    
    # Aggregate across conformers - take maximum value for each feature
    # This captures the "best" geometry for each interaction
    final_fp = np.max(all_fps, axis=0)
    
    return final_fp

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
                fingerprint = compute_fingerprint_from_mol(mol)
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
