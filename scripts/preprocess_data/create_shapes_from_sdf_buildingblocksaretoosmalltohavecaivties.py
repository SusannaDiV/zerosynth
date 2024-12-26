import os
import pickle as pkl
from shape_utils import get_atom_stamp, get_shape, ROTATIONS, centralize, get_mol_centroid, trans, get_binary_features
from tfbio_data import make_grid
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem
import copy
from random import sample
import numpy as np
import subprocess
from shape_pretraining_dataset import ShapePretrainingDataset
from tqdm.auto import tqdm
import shutil


def sdf_to_pdb(sdf_path, output_dir):
    """
    Convert each molecule in SDF to a separate PDB file with proper ATOM records.
    Generates 3D conformer if molecule is flat.
    Returns a list of generated PDB file paths.
    """
    from rdkit import Chem
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_files = []
    supplier = Chem.SDMolSupplier(sdf_path)
    
    for idx, mol in tqdm(enumerate(supplier), total=len(supplier), desc="Converting SDF to PDB"):
        if mol is not None:
            try:
                # Check if molecule is 3D
                conf = mol.GetConformer()
                positions = conf.GetPositions()
                is_2d = all(pos[2] == 0 for pos in positions)
                
                if is_2d:
                    # Generate 3D conformer
                    mol = Chem.AddHs(mol)  # Add hydrogens
                    AllChem.EmbedMolecule(mol, randomSeed=42)  # Generate 3D coords
                    AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry
                    mol = Chem.RemoveHs(mol)  # Remove hydrogens
                
                pdb_path = os.path.join(output_dir, f"molecule_{idx}.pdb")
                
                # Convert HETATM to ATOM records and standardize atom names
                pdb_lines = []
                conf = mol.GetConformer()
                
                for i, atom in enumerate(mol.GetAtoms()):
                    pos = conf.GetAtomPosition(i)
                    # Format according to PDB standard
                    line = (f"ATOM  {i+1:5d}  {atom.GetSymbol():<3}{' ':1}{'LIG':3} {'A':1}{1:4d}"
                           f"{' ':4}{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{1.00:6.2f}{0.00:6.2f}"
                           f"{' ':10}{atom.GetSymbol():>2}{' ':2}\n")
                    pdb_lines.append(line)
                
                # Add TER and END records
                pdb_lines.append("TER\nEND\n")
                
                # Write to file
                with open(pdb_path, 'w') as f:
                    f.writelines(pdb_lines)
                
                pdb_files.append(pdb_path)
            except Exception as e:
                print(f"Failed to process molecule {idx}: {e}")
                continue
            
    print(f"Converted {len(pdb_files)} molecules from {sdf_path} to PDB files")
    return pdb_files


def run_fpocket(pdb_file):
    """
    Run fpocket to detect cavities with very lenient parameters for small molecules.
    Creates and manages fpocket output directories.
    """
    # Create fpocket output directory for this specific molecule
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    mol_pocket_dir = os.path.join(fpocket_output_dir, base_name + "_out")
    
    # Clean up any existing directory
    if os.path.exists(mol_pocket_dir):
        shutil.rmtree(mol_pocket_dir)
    os.makedirs(mol_pocket_dir)
    
    # Even more lenient parameters for small molecules:
    # -m 2.0: minimum radius of alpha sphere (default 3.4)
    # -M 5.0: maximum radius of alpha sphere (default 6.2)
    # -i 10: minimum number of apolar neighbors (default 30)
    # -p 0.0: minimum proportion of apolar spheres (default 0.8)
    # -v 50: minimum pocket volume (default 100)
    # -D 1.5: maximum distance for clustering (default 2.4)
    cmd = f"fpocket -f {pdb_file} -m 2.0 -M 5.0 -i 10 -p 0.0 -v 50 -D 1.5"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        
        # fpocket creates output in input_dir_out
        fpocket_default_out = pdb_file.replace(".pdb", "_out")
        if os.path.exists(fpocket_default_out):
            # Move contents to our desired output directory
            for item in os.listdir(fpocket_default_out):
                shutil.move(
                    os.path.join(fpocket_default_out, item),
                    os.path.join(mol_pocket_dir, item)
                )
            shutil.rmtree(fpocket_default_out)
            
        print(f"fpocket analysis complete for {pdb_file}")
        return mol_pocket_dir
    except subprocess.CalledProcessError as e:
        print(f"fpocket failed for {pdb_file}: {e.stderr}")
        return None
    except Exception as e:
        print(f"Error in fpocket processing: {str(e)}")
        return None


def select_largest_pocket(fpocket_out_dir):
    """
    Select the largest pocket from fpocket output based on pocket volume.
    Returns None if no valid pockets found.
    """
    if fpocket_out_dir is None:
        return None
        
    pockets_info = os.path.join(fpocket_out_dir, "pockets.info")
    if not os.path.exists(pockets_info):
        print(f"No pockets.info file found in {fpocket_out_dir}")
        return None
        
    try:
        with open(pockets_info, 'r') as f:
            largest_pocket = None
            largest_volume = -1
            
            for line in f:
                if line.startswith("Pocket"):
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    pocket_id = parts[1]
                    try:
                        volume = float(parts[2])
                        if volume > largest_volume:
                            largest_volume = volume
                            largest_pocket = pocket_id
                    except (ValueError, IndexError):
                        continue
                        
            if largest_pocket is None:
                return None
                
            pocket_file = os.path.join(fpocket_out_dir, f"pockets/pocket{largest_pocket}_atm.pdb")
            if not os.path.exists(pocket_file):
                return None
                
            return pocket_file
            
    except Exception as e:
        print(f"Error processing fpocket output: {e}")
        return None


def is_molecule_suitable(mol):
    """Check if molecule is suitable for pocket detection"""
    if mol is None:
        return False
        
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    n_rings = mol.GetRingInfo().NumRings()
    
    # Criteria for suitability:
    # - At least 15 atoms
    # - At least 15 bonds
    # - At least 2 rings
    return n_atoms >= 15 and n_bonds >= 15 and n_rings >= 2


data_path = '/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/models/encoder/data/training_data.pkl'
input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/models/encoder/data/molecules"
fpocket_output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/models/encoder/data/allpockets"

print("Loading training data...")
with open(data_path, 'rb') as fr:
    data = pkl.load(fr)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(fpocket_output_dir, exist_ok=True)

supplier = Chem.SDMolSupplier(input_sdf)
first_5_mols = []
for i, mol in enumerate(supplier):
    if i >= 5:  
        break
    if mol is not None:
        first_5_mols.append(mol)

print(f"Processing {len(first_5_mols)} molecules...")

molecule_pdbs = []
for idx, mol in enumerate(first_5_mols):
    try:
        if not is_molecule_suitable(mol):
            print(f"Molecule {idx} too small/simple for pocket detection, skipping...")
            continue
            
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        is_2d = all(pos[2] == 0 for pos in positions)
        
        if is_2d:
            print(f"Molecule {idx} is 2D, generating 3D conformer...")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        
        pdb_path = os.path.join(output_dir, f"molecule_{idx}.pdb")
        
        # Write PDB file
        pdb_lines = []
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            line = (f"ATOM  {i+1:5d}  {atom.GetSymbol():<3}{' ':1}{'LIG':3} {'A':1}{1:4d}"
                   f"{' ':4}{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{1.00:6.2f}{0.00:6.2f}"
                   f"{' ':10}{atom.GetSymbol():>2}{' ':2}\n")
            pdb_lines.append(line)
        pdb_lines.append("TER\nEND\n")
        
        with open(pdb_path, 'w') as f:
            f.writelines(pdb_lines)
        
        molecule_pdbs.append(pdb_path)
        print(f"Successfully wrote {pdb_path}")
        
    except Exception as e:
        print(f"Failed to process molecule {idx}: {e}")
        continue

all_sample_shapes = []
all_sample_n_o_f = []

for protein_pdb in tqdm(molecule_pdbs, desc="Processing molecules"):
    try:
        # Step 2: Run fpocket for this molecule
        fpocket_out_dir = run_fpocket(protein_pdb)
        
        # Step 3: Select the largest pocket
        cavity_pdb = select_largest_pocket(fpocket_out_dir)
        if cavity_pdb is None:
            print(f"Skipping {protein_pdb} - no valid pockets found")
            continue
            
        protein_pdb_final = protein_pdb

        atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)

        # Step 5: Load cavity and protein structures
        cavity = Chem.MolFromPDBFile(cavity_pdb, proximityBonding=False)
        protein = Chem.MolFromPDBFile(protein_pdb_final, proximityBonding=False)
        
        if cavity is None or protein is None:
            print(f"Failed to load structures for {protein_pdb}")
            continue

        # Step 6: Centralize cavity and align protein
        cavity_centroid = get_mol_centroid(cavity)
        cavity = centralize(cavity)
        translation = trans(-cavity_centroid[0], -cavity_centroid[1], -cavity_centroid[2])
        protein_conformer = protein.GetConformer()
        rdMolTransforms.TransformConformer(protein_conformer, translation)

        # Step 7: Generate sample shapes and grids
        sample_shapes = []
        sample_n_o_f = []
        for i in range(200):
            print(i)
            i = i % 24

            copied_cavity = copy.deepcopy(cavity)
            copied_protein = copy.deepcopy(protein)

            cavity_conformer = copied_cavity.GetConformer()
            protein_conformer = copied_protein.GetConformer()

            rotation_mat = ROTATIONS[i]
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = rotation_mat
            rdMolTransforms.TransformConformer(cavity_conformer, rotation)
            rdMolTransforms.TransformConformer(protein_conformer, rotation)

            curr_cavity_shape = get_shape(copied_cavity, atom_stamp, 0.5, 15)
            large_cavity_shape = np.zeros((61 * 3, 61 * 3, 61 * 3))
            large_cavity_shape[61 * 1:61 * 2, 61 * 1:61 * 2, 61 * 1:61 * 2] = curr_cavity_shape

            protein_coords, protein_features = get_binary_features(copied_protein, -1, False)
            protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 15)
            protein_grid = protein_grid.squeeze()

            n_o_f_grid = np.zeros(protein_grid.shape)
            for xyz in feature_dict[(7.0,)] + feature_dict[(8.0,)] + feature_dict[(9.0,)]:
                x, y, z = xyz[0], xyz[1], xyz[2]

                x_left = x - 4 if x - 4 >= 0 else 0
                x_right = x + 4 if x + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

                y_left = y - 4 if y - 4 >= 0 else 0
                y_right = y + 4 if y + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

                z_left = z - 4 if z - 4 >= 0 else 0
                z_right = z + 4 if z + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

                tmp = n_o_f_grid[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
                tmp += 1
            large_n_o_f_grid = np.zeros((61 * 3, 61 * 3, 61 * 3))
            large_n_o_f_grid[61 * 1:61 * 2, 61 * 1:61 * 2, 61 * 1:61 * 2] = n_o_f_grid

            seed_data = sample(data, 10)
            union_shape = np.zeros((61, 61, 61))
            for seed in seed_data:
                mol_shape = get_shape(centralize(seed[0]), atom_stamp, 0.5, 15)
                union_shape = union_shape + mol_shape
            union_shape[union_shape > 1] = 1

            flag = False
            for j in range(0, 122):
                large_union_shape = np.zeros((61 * 3, 61 * 3, 61 * 3))
                large_union_shape[j: j + 61, j: j + 61, j: j + 61] = union_shape
                inter_shape = large_cavity_shape * large_union_shape
                if inter_shape.sum() > 2400:
                    flag = True
                    break
            if flag:
                inter_idx = np.where(inter_shape > 0)
                x, y, z = inter_idx[0].mean(), inter_idx[1].mean(), inter_idx[2].mean()
                x, y, z = int(x.round()), int(y.round()), int(z.round())
                x_left, x_right = x - 13, x + 14 + 1
                y_left, y_right = y - 13, y + 14 + 1
                z_left, z_right = z - 13, z + 14 + 1
                inter_shape = inter_shape[x_left: x_right, y_left: y_right, z_left: z_right]
                inter_n_o_f = large_n_o_f_grid[x_left: x_right, y_left: y_right, z_left: z_right]
                sample_shapes.append(inter_shape)
                sample_n_o_f.append(inter_n_o_f)

        sample_shapes.append(get_shape(cavity, atom_stamp, 0.5, 6.75))

        protein_coords, protein_features = get_binary_features(protein, -1, False)
        protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 6.75)
        protein_grid = protein_grid.squeeze()
        n_o_f_grid = np.zeros(protein_grid.shape)
        for xyz in feature_dict[(7.0,)] + feature_dict[(8.0,)] + feature_dict[(9.0,)]:
            x, y, z = xyz[0], xyz[1], xyz[2]

            x_left = x - 4 if x - 4 >= 0 else 0
            x_right = x + 4 if x + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

            y_left = y - 4 if y - 4 >= 0 else 0
            y_right = y + 4 if y + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

            z_left = z - 4 if z - 4 >= 0 else 0
            z_right = z + 4 if z + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

            tmp = n_o_f_grid[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
            tmp += 1
        sample_n_o_f.append(n_o_f_grid)

        all_sample_shapes.append(sample_shapes)
        all_sample_n_o_f.append(sample_n_o_f)

        print(f"Successfully processed {os.path.basename(protein_pdb)}")

    except Exception as e:
        print(f"Error processing molecule {protein_pdb}: {e}")
        continue

print(f"Saving {len(all_sample_shapes)} shapes...")
with open('/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/data/processed/all/sample_shapes.pkl', 'wb') as fw:
    pkl.dump(all_sample_shapes, fw)

processed_data = []
for molecule_shapes in all_sample_shapes:
    for shape in molecule_shapes:
        processed_data.append({'mol': shape})

dataset = ShapePretrainingDataset(
    data=processed_data,
    grid_resolution=1,
    max_dist_stamp=3.0,
    max_dist=10.0,
    patch_size=3
)

print("Saving final dataset...")
with open('/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/data/processed/all/shape_dataset.pkl', 'wb') as fw:
    pkl.dump(dataset, fw)

print("Processing complete!")