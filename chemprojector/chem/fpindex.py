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

from .mol import FingerprintOption, Molecule, read_mol_file
from .tfbio_data import get_atom_stamp, make_grid, get_binary_features, ROTATIONS

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
        self._shapes, self._shape_patches, self._ph4_patches = self._init_shapes()
        self._tree = self._init_tree()
    
    
    def get_shape_with_memory_check(cavity, atom_stamp, resolution, box_size):
        """Wrapper to check memory requirements before shape computation"""
        # Calculate required memory
        grid_points = int(2 * box_size / resolution) + 1
        required_memory = grid_points**3 * 4  # 4 bytes per float32
        
        # Check available memory (with 20% buffer)
        available_memory = psutil.virtual_memory().available * 0.8
        
        if required_memory > available_memory:
            raise MemoryError(f"Insufficient memory. Need {required_memory/1024:.0f} KiB, "
                            f"have {available_memory/1024:.0f} KiB available")
        
        return get_shape(cavity, atom_stamp, resolution, box_size)

    def process_single_rotation(self, mol, rotation_mat, atom_stamp, cavity, resolution=0.5, box_size=15):
        """Process a single rotation of a molecule with pre-computed conformers"""
        try:
            # Check inputs
            if mol is None:
                raise ValueError("Molecule is None")
            if not mol.has_conformer:
                raise ValueError(f"Molecule {mol.smiles} has no conformer")
            if cavity is None:
                raise ValueError("Cavity is None")
            if not cavity.GetNumConformers():
                raise ValueError("Cavity has no conformers")
            
            print(f"\nProcessing rotation for {mol.smiles}", flush=True)
            
            # Copy and rotate cavity
            copied_cavity = copy.deepcopy(cavity)
            if copied_cavity is None:
                raise ValueError("Failed to copy cavity")
            
            cavity_conformer = copied_cavity.GetConformer()
            if cavity_conformer is None:
                raise ValueError("Failed to get cavity conformer")
            
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = rotation_mat
            rdMolTransforms.TransformConformer(cavity_conformer, rotation)
            
            # Get shape
            print("Computing cavity shape...", flush=True)
            curr_cavity_shape = get_shape_with_memory_check(
                copied_cavity, atom_stamp, resolution, box_size
            )
            if curr_cavity_shape is None:
                raise ValueError("Failed to compute cavity shape")
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
            print(f"✓ Centered shape extracted with dimensions {centered_shape.shape}", flush=True)
            
            # Create shape patches
            shape_patches = view_as_blocks(centered_shape, (3, 3, 3))
            if shape_patches is None or shape_patches.size == 0:
                raise ValueError("Failed to create shape patches")
            shape_patches = shape_patches.reshape(-1, 27)  # 3^3 = 27
            print(f"✓ Created {len(shape_patches)} shape patches", flush=True)
            
            # Create rotated molecule for pharmacophore features
            print("Computing pharmacophore features...", flush=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rotated_mol = copy.deepcopy(mol)
            if rotated_mol is None:
                raise ValueError("Failed to copy molecule for rotation")
            
            rotated_conformer = rotated_mol._rdmol.GetConformer()
            if rotated_conformer is None:
                raise ValueError("Failed to get rotated conformer")
            
            rdMolTransforms.TransformConformer(rotated_conformer, rotation)
            
            # Get pharmacophore features
            ph4_coords, ph4_types = rotated_mol.get_pharmacophore_features(device=device)
            print(f"✓ Found {len(ph4_coords)} pharmacophore features", flush=True)
            
            # Create pharmacophore patches
            patch_size = 3 * resolution
            patches_per_dim = int(np.cbrt(len(shape_patches)))
            centers = []
            for i in range(patches_per_dim):
                for j in range(patches_per_dim):
                    for k in range(patches_per_dim):
                        center = [
                            (i + 0.5) * patch_size,
                            (j + 0.5) * patch_size,
                            (k + 0.5) * patch_size
                        ]
                        centers.append(center)
            
            patch_centers = torch.tensor(centers, dtype=torch.float32, device=device)
            ph4_patches = self.assign_ph4_to_patches(
                ph4_coords, ph4_types, patch_centers, patch_size
            )
            print(f"✓ Created pharmacophore patches", flush=True)
            
            return {
                'mol': mol,
                'shape': centered_shape,
                'shape_patches': torch.tensor(shape_patches, device=device),
                'ph4_patches': ph4_patches
            }
            
        except Exception as e:
            print(f"\nFAILED ROTATION for {mol.smiles if mol else 'None'}: {str(e)}", flush=True)
            return None

    def process_molecule_batch(self, mol_batch, atom_stamp):
        """Process a batch of molecules with parallel rotations"""
        results = []
        for mol in mol_batch:
            if mol is None:
                continue
                
            try:
                print(f"\n{'='*50}", flush=True)
                print(f"Processing molecule: {mol.smiles}", flush=True)
                
                # Generate 3D conformer with explicit checks
                rdmol = Chem.AddHs(mol._rdmol)
                if rdmol is None:
                    raise ValueError(f"Failed to add hydrogens to molecule: {mol.smiles}")
                    
                embed_result = AllChem.EmbedMolecule(rdmol, randomSeed=42)
                if embed_result == -1:  # EmbedMolecule returns -1 on failure
                    raise ValueError(f"Failed to embed 3D coordinates for molecule: {mol.smiles}")
                    
                optimize_result = AllChem.MMFFOptimizeMolecule(rdmol)
                if optimize_result == -1:  # MMFFOptimizeMolecule returns -1 on failure
                    raise ValueError(f"Failed to optimize 3D structure for molecule: {mol.smiles}")
                    
                rdmol = Chem.RemoveHs(rdmol)
                if rdmol is None:
                    raise ValueError(f"Failed to remove hydrogens from molecule: {mol.smiles}")
                    
                mol.store_conformer(rdmol)
                
                # Verify conformer was stored
                if not mol.has_conformer:
                    raise ValueError(f"Failed to store conformer for molecule: {mol.smiles}")
                    
                print("✓ Generated and stored 3D conformer", flush=True)
                
                # Create cavity
                cavity = Chem.Mol(mol._rdmol)
                if cavity is None:
                    raise ValueError(f"Failed to create cavity for molecule: {mol.smiles}")
                
                print(f"Processing {len(ROTATIONS)} rotations...", flush=True)
                
                # Process rotations
                rotation_results = []
                for i, rot_mat in enumerate(ROTATIONS):
                    print(f"\nRotation {i+1}/{len(ROTATIONS)}", flush=True)
                    result = self.process_single_rotation(mol, rot_mat, atom_stamp, cavity)
                    if result is not None:
                        rotation_results.append(result)
                        print(f"✓ Rotation {i+1} successful", flush=True)
                    else:
                        print(f"✗ Rotation {i+1} failed", flush=True)
                
                # Check results
                if not rotation_results:
                    raise ValueError(f"All rotations failed for molecule: {mol.smiles}")
                
                print(f"\n✓ Successfully processed {len(rotation_results)}/{len(ROTATIONS)} rotations", flush=True)
                results.extend(rotation_results)
                
            except Exception as e:
                print(f"\nFAILED to process molecule {mol.smiles}: {str(e)}", flush=True)
                continue
                
        if not results:
            raise ValueError(f"No molecules were successfully processed in batch of size {len(mol_batch)}")
        
        return results

    def _init_shapes(self, batch_size: int = 4) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[torch.Tensor]]]:
        shapes_dict = {}
        patches_dict = {}
        ph4_patches_dict = {}
        current_batch = []
        
        for idx in tqdm(range(0, len(self._molecules), batch_size), desc="Computing shapes"):
            mol_batch = self._molecules[idx:idx + batch_size]
            atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
            
            try:
                results = self.process_molecule_batch(mol_batch, atom_stamp)
                if not results:  # If no results were returned
                    raise ValueError(f"No results returned for batch starting at index {idx}")
                
                # Group shapes by molecule index
                for result in results:
                    mol_idx = idx + list(mol_batch).index(result['mol'])
                    if mol_idx not in shapes_dict:
                        shapes_dict[mol_idx] = []
                        patches_dict[mol_idx] = []
                        ph4_patches_dict[mol_idx] = []
                    shapes_dict[mol_idx].append(result['shape'])
                    patches_dict[mol_idx].append(result['shape_patches'])
                    ph4_patches_dict[mol_idx].append(result['ph4_patches'])
            
            except Exception as e:
                print("\n" + "="*50)
                print(f"CRITICAL ERROR in _init_shapes at batch {idx}")
                print(f"Error: {str(e)}")
                print("="*50 + "\n")
                raise  # Re-raise the exception to stop processing
            
            # Memory management
            if len(current_batch) >= 1000:
                gc.collect()
                current_batch = []
        
        return shapes_dict, patches_dict, ph4_patches_dict

    @property
    def shapes(self) -> dict[int, list[np.ndarray]]:
        return self._shapes

    @property
    def shape_patches(self) -> dict[int, list[np.ndarray]]:
        return self._shape_patches

    def get_shapes(self, index: int) -> list[np.ndarray]:
        return self._shapes.get(index, [])

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

    @property
    def ph4_patches(self) -> dict[int, list[torch.Tensor]]:
        return self._ph4_patches

    def get_ph4_patches(self, index: int) -> list[torch.Tensor]:
        return self._ph4_patches.get(index, [])

    def assign_ph4_to_patches(
        self,
        ph4_coords: torch.Tensor,  # [N_features, 3]
        ph4_types: torch.Tensor,   # [N_features]
        patch_centers: torch.Tensor,  # [num_patches, 3]
        patch_size: float,
        num_ph4_types: int = 6
    ) -> torch.Tensor:
        """
        Assign pharmacophore features to patches using exact boundary checks
        Returns: [num_patches, num_ph4_types] tensor with feature counts per patch
        """
        if len(ph4_coords) == 0:
            print("WARNING: No pharmacophore features to assign to patches")
            return torch.zeros((len(patch_centers), num_ph4_types), 
                             device=patch_centers.device)
        
        print(f"\nAssigning {len(ph4_coords)} features to {len(patch_centers)} patches")
        
        num_patches = patch_centers.size(0)
        half_size = patch_size / 2
        
        # Initialize feature counts per patch
        patch_features = torch.zeros(
            (num_patches, num_ph4_types),
            device=ph4_coords.device
        )
        
        # Calculate patch boundaries
        patch_mins = patch_centers - half_size  # [num_patches, 3]
        patch_maxs = patch_centers + half_size  # [num_patches, 3]
        
        # For each feature
        for feat_idx in range(len(ph4_coords)):
            feat_coord = ph4_coords[feat_idx]  # [3]
            feat_type = ph4_types[feat_idx]    # scalar
            
            # Check if feature is within patch bounds
            in_x = (feat_coord[0] >= patch_mins[:, 0]) & (feat_coord[0] < patch_maxs[:, 0])
            in_y = (feat_coord[1] >= patch_mins[:, 1]) & (feat_coord[1] < patch_maxs[:, 1])
            in_z = (feat_coord[2] >= patch_mins[:, 2]) & (feat_coord[2] < patch_maxs[:, 2])
            
            # Feature must be within bounds in all dimensions
            in_patch = in_x & in_y & in_z  # [num_patches]
            
            # Add feature to relevant patches
            patch_features[in_patch, feat_type] += 1
        
        # Add summary at the end
        non_empty_patches = (patch_features.sum(dim=1) > 0).sum().item()
        print(f"Features assigned to {non_empty_patches}/{len(patch_centers)} patches")
        
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

if __name__ == "__main__":
    fpindex_path = "data/processed/all/fpindex_pharmaco.pkl"
    print_first_molecule_data(fpindex_path)
