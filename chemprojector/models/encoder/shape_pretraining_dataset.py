import torch
from torch.utils.data import Dataset
import numpy as np
from math import ceil
from shape_utils import (
    get_atom_stamp, get_shape, get_shape_patches, get_atom_stamp_with_noise,
    ROTATIONS, centralize, get_mol_centroid, trans, get_binary_features
)
from tfbio_data import make_grid
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import copy
from random import sample

class ShapePretrainingDataset(Dataset):
    def __init__(self, 
                 data,
                 training_data,  # Add training_data parameter
                 grid_resolution=1.0,
                 max_dist_stamp=3.0,
                 max_dist=10.0,
                 patch_size=3,
                 shape_noise_mu=0.0,
                 shape_noise_sigma=0.0):
        self.data = data
        self.training_data = training_data
        self.grid_resolution = grid_resolution
        self.max_dist_stamp = max_dist_stamp
        self.max_dist = max_dist
        self.patch_size = patch_size
        self.shape_noise_mu = shape_noise_mu
        self.shape_noise_sigma = shape_noise_sigma
        self.box_size = ceil(2 * max_dist // grid_resolution + 1)
        self.atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)

    def process_molecule(self, mol_pdb, cavity_pdb):
        sample_shapes = []
        
        # Load structures
        cavity = Chem.MolFromPDBFile(cavity_pdb, proximityBonding=False)
        protein = Chem.MolFromPDBFile(mol_pdb, proximityBonding=False)

        # Centralize and align
        cavity_centroid = get_mol_centroid(cavity)
        cavity = centralize(cavity)
        translation = trans(-cavity_centroid[0], -cavity_centroid[1], -cavity_centroid[2])
        protein_conformer = protein.GetConformer()
        rdMolTransforms.TransformConformer(protein_conformer, translation)

        # Generate rotations and shapes
        for i in range(200):
            copied_cavity = copy.deepcopy(cavity)
            copied_protein = copy.deepcopy(protein)

            # Apply rotation
            cavity_conformer = copied_cavity.GetConformer()
            protein_conformer = copied_protein.GetConformer()
            rotation_mat = ROTATIONS[i % 24]
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = rotation_mat
            rdMolTransforms.TransformConformer(cavity_conformer, rotation)
            rdMolTransforms.TransformConformer(protein_conformer, rotation)

            # Generate shapes
            curr_cavity_shape = get_shape(copied_cavity, self.atom_stamp, 0.5, 15)
            sample_shapes.extend(self._process_shape(curr_cavity_shape, copied_protein))

        return sample_shapes

    def _process_shape(self, cavity_shape, protein):
        shapes = []
        large_cavity_shape = np.zeros((61 * 3, 61 * 3, 61 * 3))
        large_cavity_shape[61:122, 61:122, 61:122] = cavity_shape

        # Process protein features
        protein_coords, protein_features = get_binary_features(protein, -1, False)
        protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 15)
        
        # Generate union shapes from training data
        seed_data = sample(self.training_data, 10)
        union_shape = np.zeros((61, 61, 61))
        for seed in seed_data:
            mol_shape = get_shape(centralize(seed[0]), self.atom_stamp, 0.5, 15)
            union_shape = union_shape + mol_shape
        union_shape[union_shape > 1] = 1

        # Find intersections
        for j in range(0, 122):
            large_union_shape = np.zeros((61 * 3, 61 * 3, 61 * 3))
            large_union_shape[j:j + 61, j:j + 61, j:j + 61] = union_shape
            inter_shape = large_cavity_shape * large_union_shape
            if inter_shape.sum() > 2400:
                shapes.append(self._extract_intersection(inter_shape))

        return shapes

    def _extract_intersection(self, inter_shape):
        inter_idx = np.where(inter_shape > 0)
        x, y, z = map(lambda x: int(round(x.mean())), inter_idx)
        x_left, x_right = x - 13, x + 14 + 1
        y_left, y_right = y - 13, y + 14 + 1
        z_left, z_right = z - 13, z + 14 + 1
        return inter_shape[x_left:x_right, y_left:y_right, z_left:z_right]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        curr_shape = sample['mol']  # Now directly a shape
        
        curr_shape_patches = get_shape_patches(curr_shape, self.patch_size)
        curr_shape_patches = curr_shape_patches.reshape(-1, self.patch_size**3)

        return {
            'shape': torch.tensor(curr_shape, dtype=torch.long),
            'shape_patches': torch.tensor(curr_shape_patches, dtype=torch.float),
            'mol': curr_shape
        }

