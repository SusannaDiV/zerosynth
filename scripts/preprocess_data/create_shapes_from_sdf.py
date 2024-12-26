import os
import pickle as pkl
from shape_utils import get_atom_stamp, get_shape, ROTATIONS, centralize, get_mol_centroid, trans, get_binary_features
from tfbio_data import make_grid
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import copy
from random import sample
import numpy as np
import subprocess
from shape_pretraining_dataset import ShapePretrainingDataset
from tqdm.auto import tqdm


def sdf_to_pdb(sdf_path, output_dir):
    """
    Convert each molecule in SDF to a separate PDB file.
    Returns a list of generated PDB file paths.
    """
    from rdkit import Chem
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read molecules from SDF
    pdb_files = []
    supplier = Chem.SDMolSupplier(sdf_path)
    total_mols = len(supplier)  # Get total count for progress bar
    
    for idx, mol in tqdm(enumerate(supplier), total=total_mols, desc="Converting SDF to PDB"):
        if mol is not None:
            # Generate output path for this molecule
            pdb_path = os.path.join(output_dir, f"molecule_{idx}.pdb")
            
            # Write molecule to PDB
            Chem.MolToPDBFile(mol, pdb_path)
            pdb_files.append(pdb_path)
            
    print(f"Converted {len(pdb_files)} molecules from {sdf_path} to PDB files")
    return pdb_files


def run_fpocket(pdb_file):
    """
    Run fpocket to detect cavities.
    """
    cmd = f"fpocket -f {pdb_file}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"fpocket analysis complete for {pdb_file}")


def select_largest_pocket(fpocket_output_dir):
    """
    Select the largest pocket from fpocket output based on pocket volume.
    """
    pockets_info = os.path.join(fpocket_output_dir, "pockets_info.txt")
    largest_pocket = None
    largest_volume = -1
    with open(pockets_info, 'r') as f:
        for line in f:
            if line.startswith("Pocket"):
                parts = line.split()
                pocket_id = parts[1]
                volume = float(parts[2])
                if volume > largest_volume:
                    largest_volume = volume
                    largest_pocket = pocket_id
    if largest_pocket is None:
        raise ValueError("No pockets found in fpocket output.")
    pocket_file = os.path.join(fpocket_output_dir, f"/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/models/encoder/data/pockets/pocket{largest_pocket}.pdb")
    return pocket_file


# File paths
data_path = '/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/models/encoder/data/training_data.pkl'
input_sdf = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf"
output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/models/encoder/data/molecules"
fpocket_output_dir = "/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/models/encoder/data/allpockets"

print("Loading training data...")
with open(data_path, 'rb') as fr:
    data = pkl.load(fr)

# Step 1: Convert each molecule in SDF to separate PDB files
molecule_pdbs = sdf_to_pdb(input_sdf, output_dir)

# Process each molecule
all_sample_shapes = []
all_sample_n_o_f = []

for protein_pdb in tqdm(molecule_pdbs, desc="Processing molecules"):
    try:
        # Step 2: Run fpocket for this molecule
        run_fpocket(protein_pdb)
        
        # Step 3: Select the largest pocket
        largest_pocket = select_largest_pocket(fpocket_output_dir)
        cavity_pdb = largest_pocket
        protein_pdb_final = protein_pdb

        atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)

        # Step 5: Load cavity and protein structures
        cavity = Chem.MolFromPDBFile(cavity_pdb, proximityBonding=False)
        protein = Chem.MolFromPDBFile(protein_pdb_final, proximityBonding=False)

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