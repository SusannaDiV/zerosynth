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
        sparse_fp = mol.get_fingerprint(fp_option)
        # Convert sparse to dense array and ensure binary values
        dense_fp = sparse_fp.toarray()[0]
        # For E3FP float fingerprints, convert to binary
        if fp_option.type == "e3fp":
            dense_fp = (dense_fp > 0).astype(np.uint8)
        else:
            dense_fp = dense_fp.astype(np.uint8)
        fp[offset + i] = dense_fp


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
    def __init__(self, molecules: Iterable[Molecule], fp_option: FingerprintOption, encoder_type: str = None) -> None:
        super().__init__()
        self._molecules = tuple(molecules)
        self._fp_option = fp_option
        self._fp = self._init_fingerprint()
        self._shapes = None
        self._shape_patches = None
        self._tree = self._init_tree()
        
        # Only initialize shapes if using shape encoder
        if encoder_type == "shape":
            self._shapes, self._shape_patches = self._init_shapes()

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
            copied_cavity = copy.deepcopy(cavity)
            
            cavity_conformer = copied_cavity.GetConformer()
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = rotation_mat
            rdMolTransforms.TransformConformer(cavity_conformer, rotation)
            
            curr_cavity_shape = self.get_shape_with_memory_check(
                copied_cavity, atom_stamp, resolution, box_size
            )
            
            del copied_cavity
            
            grid_size = 21
            start_idx = curr_cavity_shape.shape[0]//2 - grid_size//2
            end_idx = start_idx + grid_size
            centered_shape = curr_cavity_shape[
                start_idx:end_idx,
                start_idx:end_idx,
                start_idx:end_idx
            ]
            
            shape_patches = view_as_blocks(centered_shape, (3, 3, 3))
            shape_patches = shape_patches.reshape(-1, 27)  # 3^3 = 27
            
            return {
                'mol': mol,
                'shape': centered_shape,
                'shape_patches': shape_patches
            }
            
        except Exception as e:
            print(f"Failed to process rotation: {str(e)}")
            return None

    def process_molecule_batch(self, mol_batch, atom_stamp):
        """Process a batch of molecules with parallel rotations"""
        results = []
        for mol in mol_batch:
            if mol is None:
                continue
                
            try:
                # Generate 3D conformer once
                rdmol = Chem.AddHs(mol._rdmol)
                AllChem.EmbedMolecule(rdmol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(rdmol)
                rdmol = Chem.RemoveHs(rdmol)
                mol.store_conformer(rdmol)
                
                # Create cavity using the stored conformer
                cavity = Chem.Mol(mol._rdmol)
                cavity_centroid = get_mol_centroid(cavity)
                cavity = centralize(cavity)
                
                # Process all rotations in parallel
                with mp.Pool(processes=min(24, mp.cpu_count())) as rotation_pool:
                    rotation_results = rotation_pool.starmap(
                        self.process_single_rotation,
                        [(mol, rot_mat, atom_stamp, cavity) 
                         for rot_mat in ROTATIONS]
                    )
                
                # Collect valid results
                results.extend([r for r in rotation_results if r is not None])
                
                # Clean up
                del cavity
                gc.collect()
                
            except Exception as e:
                #print(f"Failed to process molecule: {str(e)}")
                continue
                
        return results

    def _init_shapes(self, batch_size: int = 4) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
        shapes_dict = {}
        patches_dict = {}
        current_batch = []
        
        for idx in tqdm(range(0, len(self._molecules), batch_size), desc="Computing shapes"):
            mol_batch = self._molecules[idx:idx + batch_size]
            atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)
            results = self.process_molecule_batch(mol_batch, atom_stamp)
            
            # Group shapes by molecule index
            for result in results:
                mol_idx = idx + list(mol_batch).index(result['mol'])
                if mol_idx not in shapes_dict:
                    shapes_dict[mol_idx] = []
                    patches_dict[mol_idx] = []
                shapes_dict[mol_idx].append(result['shape'])
                patches_dict[mol_idx].append(result['shape_patches'])
            
            # Memory management
            if len(current_batch) >= 1000:
                gc.collect()
                current_batch = []
        
        return shapes_dict, patches_dict

    @property
    def shapes(self) -> dict[int, list[np.ndarray]]:
        if self._shapes is None:
            return {}
        return self._shapes

    @property
    def shape_patches(self) -> dict[int, list[np.ndarray]]:
        if self._shape_patches is None:
            return {}
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

def get_mol_centroid(mol):
    """Calculate the centroid of a molecule"""
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    return np.mean(positions, axis=0)

def centralize(mol):
    """Center a molecule at origin"""
    conf = mol.GetConformer()
    centroid = get_mol_centroid(mol)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, pos - centroid)
    return mol
