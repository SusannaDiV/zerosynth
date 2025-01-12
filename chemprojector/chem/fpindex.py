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
from humanize import naturalsize

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
        # self._shapes, self._shape_patches, self._ph4_patches = self._init_shapes()
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

    def process_single_rotation(self, mol, rotation_mat, atom_stamp, cavity, resolution=0.5, box_size=15, debug=False):
        """Process a single rotation of a molecule with pre-computed conformers"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Check inputs
            if mol is None:
                # raise ValueError("Molecule is None")
                return None
            if not mol.has_conformer:
                # raise ValueError(f"Molecule {mol.smiles} has no conformer")
                return None
            if cavity is None:
                # raise ValueError("Cavity is None")
                return None
            if not cavity.GetNumConformers():
                # raise ValueError("Cavity has no conformers")
                return None
            
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
            shape_patches = torch.tensor(shape_patches, device=device).to(torch.float16)
            ph4_patches = torch.tensor(ph4_patches, device=device).to(torch.float16)
            
            return {
                'mol': mol,
                #'shape': centered_shape,
                'shape_patches': shape_patches,
                'ph4_patches': ph4_patches
                # Only needed for visualization
                #'original_ph4_coords': ph4_coords if len(ph4_coords) > 0 else None
            }
            
        except Exception as e:
            if debug:
                print(f"\nFAILED ROTATION for {mol.smiles if mol else 'None'}: {str(e)}", flush=True)
            return None
        
    def process_molecule_batch(self, mol_batch, atom_stamp, debug=False):
        """Process a batch of molecules with parallel rotations"""
        results = []
        for mol in mol_batch:
            if mol is None:
                continue
                
            try:
                # Special handling for molecule 1328
                if mol == self._molecules[1328]:
                    print(f"\nSkipping problematic molecule 1328 and filling with zeros")
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    results.append({
                        'mol': mol,
                        'shape_patches': torch.zeros((7, 27), device=device).to(torch.float16),
                        'ph4_patches': torch.zeros((7, 27 * 6), device=device).to(torch.float16)
                    })
                    continue
                    
                # Normal processing for other molecules
                if debug:
                    print(f"\n{'='*50}", flush=True)
                    print(f"Processing molecule: {mol.smiles}", flush=True)
                
                # Generate 3D conformer with explicit checks
                rdmol = Chem.AddHs(mol._rdmol)
                if rdmol is None:
                    # raise ValueError(f"Failed to add hydrogens to molecule: {mol.smiles}")
                    continue
                    
                embed_result = AllChem.EmbedMolecule(rdmol, randomSeed=42)
                if embed_result == -1:  # EmbedMolecule returns -1 on failure
                    # raise ValueError(f"Failed to embed 3D coordinates for molecule: {mol.smiles}")
                    continue
                    
                optimize_result = AllChem.MMFFOptimizeMolecule(rdmol)
                if optimize_result == -1:  # MMFFOptimizeMolecule returns -1 on failure
                    # raise ValueError(f"Failed to optimize 3D structure for molecule: {mol.smiles}")
                    continue
                    
                rdmol = Chem.RemoveHs(rdmol)
                if rdmol is None:
                    # raise ValueError(f"Failed to remove hydrogens from molecule: {mol.smiles}")
                    continue
                    
                mol.store_conformer(rdmol)
                
                # Verify conformer was stored
                if not mol.has_conformer:
                    # raise ValueError(f"Failed to store conformer for molecule: {mol.smiles}")
                    continue
                    
                if debug:
                    print("✓ Generated and stored 3D conformer", flush=True)
                
                # Create cavity
                cavity = Chem.Mol(mol._rdmol)
                if cavity is None:
                    raise ValueError(f"Failed to create cavity for molecule: {mol.smiles}")
                
                if debug:
                    print(f"Processing {len(ROTATIONS)} rotations...", flush=True)
                
                rotation_results = []
                assigned_patches_counts = []
                
                for i, rot_mat in enumerate(ROTATIONS):
                    if debug:
                        print(f"\nRotation {i+1}/{len(ROTATIONS)}", flush=True)
                    result = self.process_single_rotation(mol, rot_mat, atom_stamp, cavity, debug=debug)
                    if result is not None:
                        assigned_patches = (result['ph4_patches'].sum(dim=1) > 0).sum().item()
                        assigned_patches_counts.append(assigned_patches)
                        rotation_results.append(result)
                        if debug:
                            print(f"✓ Rotation {i+1} successful", flush=True)
                    else:
                        if debug:
                            print(f"✗ Rotation {i+1} failed", flush=True)
                
                error_message = None
                try:
                    '''
                    if len(set(assigned_patches_counts)) > 1:
                        error_message = (
                            f"Inconsistent number of assigned patches across rotations for {mol.smiles}: "
                            f"counts = {assigned_patches_counts}"
                        )
                        raise ValueError(error_message)
                    '''
                    if not rotation_results:
                        error_message = f"All rotations failed for molecule: {mol.smiles}"
                        # raise ValueError(error_message)
                    
                except ValueError as e:
                    print(f"\nFAILED to process molecule {mol.smiles}: {str(e)}", flush=True)
                    # Generate visualization even for failed cases
                    # visualize_molecule_processing(mol, rotation_results, error_message)
                    continue
                
                # Generate visualization for successful cases
                # Add --visualize flag to visualize
                # visualize_molecule_processing(mol, rotation_results)
                
                if debug:
                    print(f"\n✓ Successfully processed {len(rotation_results)}/{len(ROTATIONS)} rotations", flush=True)
                results.extend(rotation_results)
                
            except Exception as e:
                if debug:
                    print(f"\nFAILED to process molecule {mol.smiles}: {str(e)}", flush=True)
                # visualize_molecule_processing(mol, [], str(e))
                continue
                
        return results

    def _init_shapes(self, batch_size: int = 4) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[torch.Tensor]]]:
        # shapes_dict = {}  # Comment out shapes dictionary
        patches_dict = {}
        ph4_patches_dict = {}
        current_batch = []
        
        for idx in tqdm(range(0, len(self._molecules), batch_size), desc="Computing shapes"):
            mol_batch = self._molecules[idx:idx + batch_size]
            atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
            
            try:
                results = self.process_molecule_batch(mol_batch, atom_stamp)
                if not results:
                    # raise ValueError(f"No results returned for batch starting at index {idx}")
                    continue
                
                # Group shapes by molecule index
                for result in results:
                    mol_idx = idx + list(mol_batch).index(result['mol'])
                    if mol_idx not in patches_dict:
                        # shapes_dict[mol_idx] = []  # Comment out shape append
                        patches_dict[mol_idx] = []
                        ph4_patches_dict[mol_idx] = []
                    # shapes_dict[mol_idx].append(result['shape'])  # Comment out shape append
                    patches_dict[mol_idx].append(result['shape_patches'])
                    ph4_patches_dict[mol_idx].append(result['ph4_patches'])
            
            except Exception as e:
                print("\n" + "="*50)
                print(f"CRITICAL ERROR in _init_shapes at batch {idx}")
                print(f"Error: {str(e)}")
                print("="*50 + "\n")
                raise
            
            # Memory management
            if len(current_batch) >= 1000:
                gc.collect()
                current_batch = []
        
        # Return empty dictionary for shapes
        # otherwise return shapes_dict, patches_dict, ph4_patches_dict
        return {}, patches_dict, ph4_patches_dict
    
    
    #@property
    #def shapes(self) -> dict[int, list[np.ndarray]]:
    #    return self._shapes
    
    #def get_shapes(self, index: int) -> list[np.ndarray]:
    #    return self._shapes.get(index, [])

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
    resume: bool = False,
    batch_size: int = 20000
):
    # Create directories for intermediate pickles
    intermediate_dir = cache_path.parent / "intermediate_pickles"
    shapes_dir = intermediate_dir / "shapes"
    ph4_dir = intermediate_dir / "ph4"
    for dir_path in [intermediate_dir, shapes_dir, ph4_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Read molecules
    all_mols = list(read_mol_file(molecule_path))
    total_mols = len(all_mols)
    print(f"\nTotal molecules to process: {total_mols}")
    
    # Find last completed batch if resuming
    start_batch = 0
    if resume:
        existing_batches = list(intermediate_dir.glob("complete_batch_*.pkl"))
        if existing_batches:
            start_batch = max(int(p.stem.split('_')[-1]) for p in existing_batches) + 1
            processed_mols = start_batch * batch_size
            print(f"\nResuming from batch {start_batch}")
            print(f"Already processed: {processed_mols} molecules")
    
    # Process remaining batches
    for batch_idx, start_idx in enumerate(range(start_batch * batch_size, total_mols, batch_size), start=start_batch):
        end_idx = min(start_idx + batch_size, total_mols)
        mol_batch = all_mols[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx} ({start_idx} to {end_idx})...")
        
        # Create index for this batch
        fpindex = FingerprintIndex(mol_batch, fp_option=fp_option)
        
        # Generate shapes and patches
        print("Generating shapes and patches...")
        _, shape_patches, ph4_patches = fpindex._init_shapes()
        
        # Save intermediate pickles separately
        shapes_path = shapes_dir / f"shapes_batch_{batch_idx:03d}.pkl"
        ph4_path = ph4_dir / f"ph4_batch_{batch_idx:03d}.pkl"
        main_path = intermediate_dir / f"main_batch_{batch_idx:03d}.pkl"
        complete_path = intermediate_dir / f"complete_batch_{batch_idx:03d}.pkl"
        
        print(f"\nSaving batch files...")
        print(f"- Shapes: {shapes_path}")
        print(f"- Ph4: {ph4_path}")
        print(f"- Main: {main_path}")
        print(f"- Complete: {complete_path}")
        
        # Save individual components
        with open(shapes_path, "wb") as f:
            pickle.dump(shape_patches, f)
        
        with open(ph4_path, "wb") as f:
            pickle.dump(ph4_patches, f)
            
        with open(main_path, "wb") as f:
            pickle.dump({
                'molecules': mol_batch,
                'fp': fpindex._fp,
                'fp_option': fp_option
            }, f)
            
        # Save complete batch
        with open(complete_path, "wb") as f:
            pickle.dump({
                'molecules': mol_batch,
                'fp': fpindex._fp,
                'fp_option': fp_option,
                'shape_patches': shape_patches,
                'ph4_patches': ph4_patches
            }, f)
        
        # Verify saved files
        print("\nVerifying saved batch files:")
        
        '''       
        # Verify shapes
        with open(shapes_path, "rb") as f:
            loaded_shapes = pickle.load(f)
            num_shapes = len(loaded_shapes)
            num_rotations = sum(len(patches) for patches in loaded_shapes.values())
            print(f"- Shape patches: {num_shapes} molecules, {num_rotations} total rotations")
            if num_shapes == 0:
                print("  WARNING: No shape patches saved!")
        
        # Verify ph4
        with open(ph4_path, "rb") as f:
            loaded_ph4 = pickle.load(f)
            num_ph4 = len(loaded_ph4)
            num_rotations = sum(len(patches) for patches in loaded_ph4.values())
            print(f"- Ph4 patches: {num_ph4} molecules, {num_rotations} total rotations")
            if num_ph4 == 0:
                print("  WARNING: No ph4 patches saved!")
        # Verify main
        with open(main_path, "rb") as f:
            loaded_main = pickle.load(f)
            print(f"- Main file: {len(loaded_main['molecules'])} molecules")
            print(f"  Fingerprint shape: {loaded_main['fp'].shape}")
        
        # Verify complete file
        with open(complete_path, "rb") as f:
            loaded_complete = pickle.load(f)
            print(f"- Complete file:")
            print(f"  - Molecules: {len(loaded_complete['molecules'])}")
            print(f"  - Fingerprint shape: {loaded_complete['fp'].shape}")
            print(f"  - Shape patches: {len(loaded_complete['shape_patches'])} molecules")
            print(f"  - Ph4 patches: {len(loaded_complete['ph4_patches'])} molecules")
            
            # Verify patch counts match
            if len(loaded_complete['shape_patches']) != len(loaded_shapes):
                print("  WARNING: Shape patch counts don't match between files!")
            if len(loaded_complete['ph4_patches']) != len(loaded_ph4):
                print("  WARNING: Ph4 patch counts don't match between files!")
        '''

        # Memory cleanup
        del fpindex, shape_patches, ph4_patches
        # del loaded_shapes, loaded_ph4, loaded_main, loaded_complete
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\nBatch processing complete!")
        
    # Merge all intermediate pickles
    print("\nMerging intermediate pickles...")
    all_molecules = []
    all_shape_patches = {}
    all_ph4_patches = {}
    all_fingerprints = []
    
    for batch_idx in range(len(list(intermediate_dir.glob("main_batch_*.pkl")))):
        print(f"\nLoading batch {batch_idx}...")
        
        # Load main data
        with open(intermediate_dir / f"main_batch_{batch_idx:03d}.pkl", "rb") as f:
            main_data = pickle.load(f)
            
        # Load shapes
        with open(shapes_dir / f"shapes_batch_{batch_idx:03d}.pkl", "rb") as f:
            shape_patches = pickle.load(f)
            
        # Load ph4
        with open(ph4_dir / f"ph4_batch_{batch_idx:03d}.pkl", "rb") as f:
            ph4_patches = pickle.load(f)
            
        # Update indices
        offset = len(all_molecules)
        shape_patches = {k + offset: v for k, v in shape_patches.items()}
        ph4_patches = {k + offset: v for k, v in ph4_patches.items()}
        
        # Extend/update collections
        all_molecules.extend(main_data['molecules'])
        all_shape_patches.update(shape_patches)
        all_ph4_patches.update(ph4_patches)
        all_fingerprints.append(main_data['fp'])
    
    # Verify pickle contains patches
    print("\nVerifying saved pickle:")
    with open(cache_path, "rb") as f:
        loaded = pickle.load(f)
        print(f"Shape patches in pickle: {len(loaded._shape_patches)}")
        print(f"Ph4 patches in pickle: {len(loaded._ph4_patches)}")
    
    # Additionally save separate patch files
    shape_patches_path = cache_path.parent / (cache_path.stem + "_shape_patches.pkl")
    ph4_patches_path = cache_path.parent / (cache_path.stem + "_ph4_patches.pkl")
    
    print(f"\nSaving additional patch files:")
    print(f"Shape patches -> {shape_patches_path}")
    print(f"Ph4 patches -> {ph4_patches_path}")
    
    with open(shape_patches_path, "wb") as f:
        pickle.dump(fpindex._shape_patches, f)
    
    with open(ph4_patches_path, "wb") as f:
        pickle.dump(fpindex._ph4_patches, f)
    
    return fpindex

def print_first_molecule_data(fpindex_path: str):
    """Print all data for the first molecule in the FingerprintIndex"""
    import pickle
    import numpy as np
    import torch
    
    print(f"\nLoading FingerprintIndex from {fpindex_path}...")
    with open(fpindex_path, 'rb') as f:
        fpindex = pickle.load(f)
    
    # Print object structure
    print("\nFingerprintIndex attributes:")
    for attr in dir(fpindex):
        if not attr.startswith('__'):  # Skip built-in attributes
            print(f"- {attr}")
    
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
    patches = fpindex._shape_patches.get(0, [])
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
    fpindex_path = "data/processed/all/fpindex_pharmacomit.pkl"
    print_first_molecule_data(fpindex_path)
