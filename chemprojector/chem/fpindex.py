import dataclasses
import functools
import os
import pathlib
import pickle
import tempfile
from collections.abc import Iterable, Sequence
import copy
import joblib
import numpy as np
import torch
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm
import psutil
import gc
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem
from skimage.util import view_as_blocks
import h5py

from .mol import FingerprintOption, Molecule, read_mol_file
from .tfbio_data import get_atom_stamp, make_grid, get_binary_features, ROTATIONS, get_shape

@dataclasses.dataclass
class _QueryResult:
    index: int
    molecule: Molecule
    fingerprint: np.ndarray
    distance: float


def _fill_fingerprint(
    fp: np.memmap,
    offset: int,
    molecules: Iterable[Molecule],
    fp_option: FingerprintOption,
):
    os.sched_setaffinity(0, range(os.cpu_count() or 1))
    for i, mol in enumerate(molecules):
        fp[offset + i] = mol.get_fingerprint(fp_option).astype(np.uint8)


def compute_fingerprints(
    molecules: Sequence[Molecule],
    fp_option: FingerprintOption,
    batch_size: int = 1024,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tempdir_s:
        temp_fname = pathlib.Path(tempdir_s) / "fingerprint"
        fp = np.memmap(
            str(temp_fname),
            dtype=np.uint8,
            mode="w+",
            shape=(len(molecules), fp_option.dim),
        )
        joblib.Parallel(n_jobs=joblib.cpu_count() // 2)(
            joblib.delayed(_fill_fingerprint)(
                fp=fp,
                offset=start,
                molecules=molecules[start : start + batch_size],
                fp_option=fp_option,
            )
            for start in tqdm(range(0, len(molecules), batch_size), desc="Fingerprint")
        )
        return np.array(fp)


class FingerprintIndex:
    def __init__(self, molecules: Iterable[Molecule], fp_option: FingerprintOption) -> None:
        super().__init__()
        self._molecules = tuple(molecules)
        self._fp_option = fp_option
        self._fp = self._init_fingerprint()
        self._tree = self._init_tree()
        
        # Store paths instead of actual data
        self.base_path = "data/processed/all"
        self.shapes_path = f"{self.base_path}/shapes_patches.h5"
        self.ph4_path = f"{self.base_path}/ph4_patches.h5"
        
        # Initialize shapes and patches
        os.makedirs(self.base_path, exist_ok=True)
        self._init_shapes()

    def verify_result_structure(self, result, mol_idx):
        """Verify that result contains all required data with correct shapes"""
        required_keys = ['shape', 'shape_patches', 'ph4_patches', 'shape_indices', 'ph4_indices']
        
        # Check all required keys exist
        if not all(key in result for key in required_keys):
            missing = [key for key in required_keys if key not in result]
            print(f"Molecule {mol_idx}: Missing required keys: {missing}")
            return False
        
        # Verify rotations consistency
        n_rotations = len(result['shape'])
        if not all(len(result[key]) == n_rotations for key in ['shape_patches', 'ph4_patches', 'shape_indices', 'ph4_indices']):
            print(f"Molecule {mol_idx}: Inconsistent number of rotations across arrays")
            return False
        
        # Verify shapes match for each rotation
        for rot_idx in range(n_rotations):
            shape = result['shape'][rot_idx]
            shape_patches = result['shape_patches'][rot_idx]
            ph4_patches = result['ph4_patches'][rot_idx]
            
            if not isinstance(shape, (torch.Tensor, np.ndarray)):
                print(f"Molecule {mol_idx}, Rotation {rot_idx}: Shape is not a tensor/array")
                return False
            
            if not isinstance(shape_patches, torch.Tensor) or not isinstance(ph4_patches, torch.Tensor):
                print(f"Molecule {mol_idx}, Rotation {rot_idx}: Patches are not tensors")
                return False
            
            if shape_patches.shape[0] != ph4_patches.shape[0]:
                print(f"Molecule {mol_idx}, Rotation {rot_idx}: Mismatched number of patches")
                return False
        
        return True

    def process_molecule_batch(self, mol_batch, atom_stamp, debug=False):
        """Process a batch of molecules with parallel rotations"""
        results = []
        
        for mol in mol_batch:
            if mol is None:
                continue
            
            try:
                # Generate 3D conformer
                rdmol = Chem.AddHs(mol._rdmol)
                if rdmol is None:
                    raise ValueError("Failed to add hydrogens")
                
                embed_result = AllChem.EmbedMolecule(rdmol, randomSeed=42)
                if embed_result == -1:
                    raise ValueError("Failed to embed 3D coordinates")
                
                optimize_result = AllChem.MMFFOptimizeMolecule(rdmol)
                if optimize_result == -1:
                    raise ValueError("Failed to optimize 3D structure")
                
                rdmol = Chem.RemoveHs(rdmol)
                mol.store_conformer(rdmol)
                
                # Process rotations
                rotation_results = []
                shape_indices = []
                ph4_indices = []
                
                for rot_idx, rot_mat in enumerate(ROTATIONS):
                    try:
                        result = self.process_single_rotation(mol, rot_mat, atom_stamp)
                        if result is None:
                            print(f"Warning: Rotation {rot_idx} failed for molecule")
                            continue
                        
                        # Get non-empty patch indices
                        shape_patches = result['shape_patches']
                        ph4_patches = result['ph4_patches']
                        
                        shape_mask = (shape_patches.sum(dim=(1,2)) > 0)
                        ph4_mask = (ph4_patches.sum(dim=(1,2)) > 0)
                        
                        # Store indices
                        result['shape_indices'] = torch.where(shape_mask)[0]
                        result['ph4_indices'] = torch.where(ph4_mask)[0]
                        
                        rotation_results.append(result)
                        shape_indices.append(result['shape_indices'])
                        ph4_indices.append(result['ph4_indices'])
                        
                    except Exception as e:
                        print(f"Failed processing rotation {rot_idx}: {str(e)}")
                        continue
                
                if not rotation_results:
                    print("No successful rotations for molecule")
                    continue
                
                # Combine results
                combined_result = {
                    'mol': mol,
                    'shape': [r['shape'] for r in rotation_results],
                    'shape_patches': torch.stack([r['shape_patches'] for r in rotation_results]),
                    'ph4_patches': torch.stack([r['ph4_patches'] for r in rotation_results]),
                    'shape_indices': shape_indices,
                    'ph4_indices': ph4_indices
                }
                
                results.append(combined_result)
                
            except Exception as e:
                print(f"Failed to process molecule: {str(e)}")
                continue
            
        return results

    def _init_shapes(self, batch_size: int = 4) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[torch.Tensor]]]:
        """Initialize shape and pharmacophore feature computation"""
        print("\nInitializing shape computation...")
        
        # Define HDF5 paths as class properties
        self.shapes_path = "shapes.h5"
        self.ph4_path = "pharmacophores.h5"
        
        # Create atom stamp once for all batches
        atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
        
        # Track successful saves
        successful_saves = 0
        
        with h5py.File(self.shapes_path, 'w') as f_shapes, h5py.File(self.ph4_path, 'w') as f_ph4:
            # Add metadata
            f_shapes.attrs['num_molecules'] = len(self._molecules)
            f_ph4.attrs['num_molecules'] = len(self._molecules)
            
            for idx in tqdm(range(0, len(self._molecules), batch_size), desc="Computing shapes"):
                mol_batch = self._molecules[idx:idx + batch_size]
                
                try:
                    results = self.process_molecule_batch(mol_batch, atom_stamp)
                    if not results:
                        continue
                    
                    # Store results in HDF5
                    for result in results:
                        mol_idx = idx + list(mol_batch).index(result['mol'])
                        mol_id = str(mol_idx)
                        
                        # Verify result structure before saving
                        if not self.verify_result_structure(result, mol_idx):
                            print(f"Skipping molecule {mol_idx} due to invalid data structure")
                            continue
                        
                        try:
                            # Store shape data
                            if mol_id not in f_shapes:
                                shape_group = f_shapes.create_group(mol_id)
                                for rot_idx in range(len(result['shape'])):
                                    rot_group = shape_group.create_group(f'rotation_{rot_idx}')
                                    
                                    # Save shape data with compression
                                    rot_group.create_dataset('shape', 
                                                           data=result['shape'][rot_idx].cpu().numpy(),
                                                           compression='gzip')
                                    rot_group.create_dataset('patches', 
                                                           data=result['shape_patches'][rot_idx].cpu().numpy(),
                                                           compression='gzip')
                                    rot_group.create_dataset('indices', 
                                                           data=result['shape_indices'][rot_idx].cpu().numpy(),
                                                           compression='gzip')
                            
                            # Store pharmacophore data
                            if mol_id not in f_ph4:
                                ph4_group = f_ph4.create_group(mol_id)
                                for rot_idx in range(len(result['ph4_patches'])):
                                    rot_group = ph4_group.create_group(f'rotation_{rot_idx}')
                                    
                                    # Save ph4 data with compression
                                    rot_group.create_dataset('patches', 
                                                           data=result['ph4_patches'][rot_idx].cpu().numpy(),
                                                           compression='gzip')
                                    rot_group.create_dataset('indices', 
                                                           data=result['ph4_indices'][rot_idx].cpu().numpy(),
                                                           compression='gzip')
                                
                                # Store fingerprint reference
                                if hasattr(self, '_fp') and mol_idx < len(self._fp):
                                    ph4_group.create_dataset('fingerprint_idx', data=mol_idx)
                            
                            successful_saves += 1
                            
                        except Exception as e:
                            print(f"Failed to save molecule {mol_idx} to HDF5: {str(e)}")
                            continue
                        
                    # Memory management
                    if idx % 100 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                except Exception as e:
                    print(f"Error processing batch at index {idx}: {str(e)}")
                    continue
            
            print(f"\nProcessing completed!")
            print(f"Successfully saved {successful_saves} molecules")
            print(f"Molecules in shapes HDF5: {len(f_shapes.keys())}")
            print(f"Molecules in ph4 HDF5: {len(f_ph4.keys())}")

        return {}, {}, {}  # Return empty dicts since data is in HDF5

    def process_single_rotation(self, mol, rotation_mat, atom_stamp, cavity, resolution=0.5, box_size=15, debug=False):
        """Process a single rotation of a molecule with pre-computed conformers"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Check inputs
            if mol is None:
                raise ValueError("Molecule is None")
            if not mol.has_conformer:
                raise ValueError(f"Molecule {mol.smiles} has no conformer")
            if cavity is None:
                raise ValueError("Cavity is None")
            if not cavity.GetNumConformers():
                raise ValueError("Cavity has no conformers")
            
            if debug:
                print(f"\nProcessing rotation for {mol.smiles}", flush=True)
            
            # Create rotation matrices
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = rotation_mat
            rotation_tensor = torch.tensor(rotation[:3, :3], device=device, dtype=torch.float32)
            
            # Get pharmacophore features and rotate
            if debug:
                print("Computing pharmacophore features...", flush=True)
            ph4_coords, ph4_types = mol.get_pharmacophore_features(device=device)
            if debug:
                print(f"✓ Found {len(ph4_coords)} pharmacophore features", flush=True)
            
            if len(ph4_coords) > 0:
                ph4_coords = torch.matmul(ph4_coords, rotation_tensor.T)
            
            # Transform to grid space with more precise binning
            grid_size = int(2 * box_size / resolution) + 1
            
            # Round coordinates to a fixed precision before binning
            grid_coords = (ph4_coords + box_size) / resolution
            grid_coords = torch.round(grid_coords * 1000) / 1000  # Round to 3 decimal places
            bin_indices = grid_coords.floor().long()  # Consistent floor operation
            
            # Create grid and assign features
            ph4_grid = torch.zeros((grid_size, grid_size, grid_size, 6), device=device)
            
            if debug:
                print("\nDEBUG Binning:")
            for feat_idx in range(len(ph4_coords)):
                x, y, z = bin_indices[feat_idx]
                feat_type = ph4_types[feat_idx]
                
                if debug:
                    print(f"Feature {feat_idx}:")
                    print(f"  Original coords: {ph4_coords[feat_idx]}")
                    print(f"  Grid coords (rounded): {grid_coords[feat_idx]}")
                    print(f"  Bin indices: {bin_indices[feat_idx]}")
                
                if (0 <= x < grid_size) and (0 <= y < grid_size) and (0 <= z < grid_size):
                    ph4_grid[x, y, z, feat_type] += 1
            
            # Process cavity and get shape
            if debug:
                print("Computing cavity shape...", flush=True)
            
            copied_cavity = self._copy_rdmol_with_conformer(cavity)
            cavity_conformer = copied_cavity.GetConformer()
            rdMolTransforms.TransformConformer(cavity_conformer, rotation)
            
            curr_cavity_shape = get_shape(copied_cavity, atom_stamp, resolution, box_size)
            if curr_cavity_shape is None:
                raise ValueError("Failed to compute cavity shape")
            
            if debug:
                print(f"✓ Shape computed with dimensions {curr_cavity_shape.shape}", flush=True)
            
            del copied_cavity
            
            # Center and extract shape
            grid_size = 21
            start_idx = curr_cavity_shape.shape[0]//2 - grid_size//2
            end_idx = start_idx + grid_size
            
            centered_shape = curr_cavity_shape[
                start_idx:end_idx,
                start_idx:end_idx,
                start_idx:end_idx
            ]
            if debug:
                print(f"✓ Centered shape extracted with dimensions {centered_shape.shape}", flush=True)
            
            centered_ph4 = ph4_grid[
                start_idx:end_idx,
                start_idx:end_idx,
                start_idx:end_idx
            ]
            
            # Convert to numpy for view_as_blocks
            centered_shape_np = centered_shape.cpu().numpy() if isinstance(centered_shape, torch.Tensor) else centered_shape
            centered_ph4_np = centered_ph4.cpu().numpy()
            
            # Create patches
            shape_patches = view_as_blocks(centered_shape_np, (3, 3, 3))
            shape_patches = shape_patches.reshape(-1, 27)
            
            ph4_patches = view_as_blocks(centered_ph4_np, (3, 3, 3, 6))
            ph4_patches = ph4_patches.reshape(-1, 27 * 6)
            
            if debug:
                print(f"✓ Created {len(shape_patches)} shape patches")
                print(f"✓ Created {len(shape_patches)} shape patches and corresponding ph4 patches")
            
            # Convert back to torch tensors
            shape_patches = torch.tensor(shape_patches, device=device)
            ph4_patches = torch.tensor(ph4_patches, device=device)
            
            return {
                'mol': mol,
                'shape': centered_shape,
                'shape_patches': shape_patches,
                'ph4_patches': ph4_patches,
                'original_ph4_coords': ph4_coords if len(ph4_coords) > 0 else None
            }
            
        except Exception as e:
            if debug:
                print(f"\nFAILED ROTATION for {mol.smiles if mol else 'None'}: {str(e)}", flush=True)
            return None
        
    def _copy_rdmol_with_conformer(self, rdmol: Chem.Mol) -> Chem.Mol:
        """Helper function to copy an RDKit molecule with its conformer."""
        if rdmol.GetNumConformers() == 0:
            raise ValueError("Input molecule has no conformers")
        
        # Create new molecule
        new_mol = Chem.Mol(rdmol)
        
        # Copy conformer explicitly
        conf = rdmol.GetConformer()
        new_mol.AddConformer(conf)
        
        return new_mol

    @property
    def molecules(self) -> tuple[Molecule, ...]:
        return self._molecules

    @property
    def fp_option(self) -> FingerprintOption:
        return self._fp_option

    def _init_fingerprint(self, batch_size: int = 1024) -> np.ndarray:
        return compute_fingerprints(
            molecules=self._molecules,
            fp_option=self._fp_option,
            batch_size=batch_size,
        )

    def _init_tree(self) -> BallTree:
        tree = BallTree(self._fp, metric="manhattan")
        return tree

    def __getitem__(self, index: int) -> tuple[Molecule, np.ndarray]:
        return self._molecules[index], self._fp[index]

    def query(self, q: np.ndarray, k: int) -> list[list[_QueryResult]]:
        """
        Args:
            q: shape (bsz, ..., fp_dim)
        """
        bsz = q.shape[0]
        dist, idx = self._tree.query(q.reshape([-1, self._fp_option.dim]), k=k)
        dist = dist.reshape([bsz, -1])
        idx = idx.reshape([bsz, -1])
        results: list[list[_QueryResult]] = []
        for i in range(dist.shape[0]):
            res: list[_QueryResult] = []
            for j in range(dist.shape[1]):
                index = int(idx[i, j])
                res.append(
                    _QueryResult(
                        index=index,
                        molecule=self._molecules[index],
                        fingerprint=self._fp[index],
                        distance=dist[i, j],
                    )
                )
            results.append(res)
        return results

    @functools.cache
    def fp_cuda(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._fp, dtype=torch.float, device=device)

    @torch.inference_mode()
    def query_cuda(self, q: torch.Tensor, k: int) -> list[list[_QueryResult]]:
        bsz = q.size(0)
        q = q.reshape([-1, self._fp_option.dim])
        pwdist = torch.cdist(self.fp_cuda(q.device), q, p=1)  # (n_mols, n_queries)
        dist_t, idx_t = torch.topk(pwdist, k=k, dim=0, largest=False)  # (k, n_queries)
        dist = dist_t.t().reshape([bsz, -1]).cpu().numpy()
        idx = idx_t.t().reshape([bsz, -1]).cpu().numpy()

        results: list[list[_QueryResult]] = []
        for i in range(dist.shape[0]):
            res: list[_QueryResult] = []
            for j in range(dist.shape[1]):
                index = int(idx[i, j])
                res.append(
                    _QueryResult(
                        index=index,
                        molecule=self._molecules[index],
                        fingerprint=self._fp[index],
                        distance=dist[i, j],
                    )
                )
            results.append(res)
        return results

    def assign_ph4_to_patches(
        self,
        ph4_coords: torch.Tensor,  # [N_features, 3]
        ph4_types: torch.Tensor,   # [N_features]
        patch_centers: torch.Tensor,  # [num_patches, 3]
        patch_size: float,
        num_ph4_types: int = 6
    ) -> torch.Tensor:
        """
        Assign pharmacophore features to patches based on spatial location
        """
        if len(ph4_coords) == 0:
            print("WARNING: No pharmacophore features to assign to patches")
            return torch.zeros((len(patch_centers), num_ph4_types), 
                             device=patch_centers.device)
        
        print(f"\nDEBUG: Feature coordinate range:")
        print(f"X: [{ph4_coords[:, 0].min():.2f}, {ph4_coords[:, 0].max():.2f}]")
        print(f"Y: [{ph4_coords[:, 1].min():.2f}, {ph4_coords[:, 1].max():.2f}]")
        print(f"Z: [{ph4_coords[:, 2].min():.2f}, {ph4_coords[:, 2].max():.2f}]")
        
        print(f"\nDEBUG: Patch center range:")
        print(f"X: [{patch_centers[:, 0].min():.2f}, {patch_centers[:, 0].max():.2f}]")
        print(f"Y: [{patch_centers[:, 1].min():.2f}, {patch_centers[:, 1].max():.2f}]")
        print(f"Z: [{patch_centers[:, 2].min():.2f}, {patch_centers[:, 2].max():.2f}]")
        
        print(f"\nAssigning {len(ph4_coords)} features to {len(patch_centers)} patches")
        
        num_patches = patch_centers.size(0)
        half_size = patch_size / 2
        
        # Initialize feature counts per patch
        patch_features = torch.zeros(
            (num_patches, num_ph4_types),
            device=ph4_coords.device
        )
        
        # For each feature
        for feat_idx in range(len(ph4_coords)):
            feat_coord = ph4_coords[feat_idx]
            feat_type = ph4_types[feat_idx]
            
            # Find patches where feature is within bounds
            in_x = (feat_coord[0] >= patch_centers[:, 0] - half_size) & (feat_coord[0] < patch_centers[:, 0] + half_size)
            in_y = (feat_coord[1] >= patch_centers[:, 1] - half_size) & (feat_coord[1] < patch_centers[:, 1] + half_size)
            in_z = (feat_coord[2] >= patch_centers[:, 2] - half_size) & (feat_coord[2] < patch_centers[:, 2] + half_size)
            in_patch = in_x & in_y & in_z
            
            matching_patches = torch.where(in_patch)[0]
            patch_features[matching_patches, feat_type] += 1
        
        assigned_patches = (patch_features.sum(dim=1) > 0).sum().item()
        print(f"Features assigned to {assigned_patches}/{num_patches} patches")
        
        return patch_features


def create_fingerprint_index_cache(
    molecule_path: pathlib.Path,
    cache_path: pathlib.Path,
    fp_option: FingerprintOption,
):
    mols = list(read_mol_file(molecule_path))
    fpindex = FingerprintIndex(mols, fp_option=fp_option)
    with open(cache_path, "wb") as f:
        pickle.dump(fpindex, f)
    return fpindex

def print_first_molecule_data(fpindex_path: str):
    """Print all data for the first molecule in the FingerprintIndex"""
    import pickle
    import numpy as np
    import torch
    
    print(f"\nLoading FingerprintIndex from {fpindex_path}...")
    with open(fpindex_path, 'rb') as f:
        fpindex = pickle.load(f)
    
    # Get first molecule
    mol = fpindex.molecules[0]
    print("\n" + "="*50)
    print("FIRST MOLECULE DATA")
    print("="*50)
    
    # Basic molecule info
    print(f"\nSMILES: {mol.smiles}")
    print(f"Has conformer: {mol._rdmol.GetNumConformers() > 0}")
    
    # Fingerprint
    fp = fpindex._fp[0]
    print(f"\nFingerprint (first 10 bits): {fp[:10]}")
    print(f"Fingerprint shape: {fp.shape}")
    
    # Shapes
    shapes = fpindex.get_shapes(0)
    if shapes:
        print(f"\nNumber of shape rotations: {len(shapes)}")
        print(f"Shape grid dimensions: {shapes[0].shape}")
        print(f"First shape values (min, max, mean): {np.min(shapes[0]):.3f}, {np.max(shapes[0]):.3f}, {np.mean(shapes[0]):.3f}")
    else:
        print("\nNo shapes found!")
    
    # Shape patches
    patches = fpindex.shape_patches.get(0, [])
    if patches:
        print(f"\nNumber of patch rotations: {len(patches)}")
        print(f"Patches per rotation: {patches[0].shape}")
        print(f"First patch rotation values (min, max, mean): {torch.min(patches[0]):.3f}, {torch.max(patches[0]):.3f}, {torch.mean(patches[0]):.3f}")
    else:
        print("\nNo shape patches found!")
    
    # Pharmacophore patches
    ph4_patches = fpindex.get_ph4_patches(0)
    if ph4_patches:
        print(f"\nNumber of ph4 patch rotations: {len(ph4_patches)}")
        print(f"Ph4 patches dimensions: {ph4_patches[0].shape}")
        print("\nPh4 feature counts in first rotation:")
        for feature_idx in range(ph4_patches[0].shape[1]):
            count = ph4_patches[0][:, feature_idx].sum().item()
            print(f"Feature type {feature_idx}: {count} total occurrences")
    else:
        print("\nNo pharmacophore patches found!")
    
    print("\n" + "="*50)

def visualize_molecule_processing(
    mol, 
    rotation_results, 
    error_message=None, 
    save_path="shapes"
):
    """
    Generate a visualization of molecule processing results including 3D shapes and pharmacophore features
    """
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    import pathlib
    import datetime
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Create save directory if it doesn't exist
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(25, 25))
    
    # Add title with SMILES and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    title = f"Molecule: {mol.smiles}\n{timestamp}"
    if error_message:
        title += f"\nERROR: {error_message}"
    plt.suptitle(title, fontsize=10, wrap=True)
    
    # 1. Original molecule with pharmacophore features
    ax1 = plt.subplot(5, 1, 1)
    img = Draw.MolToImage(mol._rdmol)
    ax1.imshow(img)
    ax1.set_title("Original Molecule")
    ax1.axis('off')
    
    if rotation_results:
        n_rotations = len(rotation_results)
        
        # 2. 3D pharmacophore feature positions for each rotation
        for i, result in enumerate(rotation_results):
            ax = plt.subplot(5, n_rotations, n_rotations + i + 1, projection='3d')
            
            # Get original 3D coordinates of pharmacophore features
            ph4_coords = result.get('original_ph4_coords', None)
            if ph4_coords is not None and len(ph4_coords) > 0:
                # Convert to numpy if needed
                coords = ph4_coords.cpu().numpy() if isinstance(ph4_coords, torch.Tensor) else ph4_coords
                
                # Plot 3D points
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                          c='red', s=100, alpha=0.8)
                
                # Add connecting lines between points
                for j in range(len(coords)):
                    for k in range(j+1, len(coords)):
                        ax.plot([coords[j,0], coords[k,0]], 
                               [coords[j,1], coords[k,1]], 
                               [coords[j,2], coords[k,2]], 
                               'r-', alpha=0.2)
            
            ax.set_title(f"Rotation {i+1}\n3D Ph4 Features")
            
            # Set equal aspect ratio and remove labels
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            
            # Set consistent viewing angle
            ax.view_init(elev=20, azim=45)
        
        # Rest of the visualization remains the same...
        # 3. 3D shape visualization
        for i, result in enumerate(rotation_results):
            shape = result['shape']
            ax = plt.subplot(5, n_rotations, 2*n_rotations + i + 1, projection='3d')
            
            shape_np = shape.cpu().numpy() if isinstance(shape, torch.Tensor) else shape
            x, y, z = np.where(shape_np > 0.5)
            scatter = ax.scatter(x, y, z, c=shape_np[x, y, z], cmap='viridis', alpha=0.6)
            ax.set_title(f"Rotation {i+1}\n3D Shape")
            
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(elev=20, azim=45)
        
        # 4. Shape patches
        for i, result in enumerate(rotation_results):
            shape_patches = result['shape_patches']
            non_empty = (shape_patches.sum(dim=1) > 0)
            patches_to_show = shape_patches[non_empty][:5]
            
            plt.subplot(5, n_rotations, 3*n_rotations + i + 1)
            plt.imshow(patches_to_show.cpu().numpy())
            plt.title(f"Rotation {i+1}\nShape Patches")
            plt.axis('off')
        
        # 5. Pharmacophore patches
        for i, result in enumerate(rotation_results):
            ph4_patches = result['ph4_patches']
            non_empty = (ph4_patches.sum(dim=1) > 0)
            patches_to_show = ph4_patches[non_empty][:5]
            
            plt.subplot(5, n_rotations, 4*n_rotations + i + 1)
            plt.imshow(patches_to_show.cpu().numpy())
            plt.title(f"Rotation {i+1}\nPh4 Patches")
            plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"{mol.smiles[:30]}_{timestamp}.png"
    filename = "".join(c if c.isalnum() else "_" for c in filename)
    plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    fpindex_path = "data/processed/all/shape_pharmaco_separate_20250110_150000.pkl"
    print_first_molecule_data(fpindex_path)
