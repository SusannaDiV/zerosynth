import dataclasses
import hashlib
import os
import pathlib
from collections.abc import Iterable, Sequence
from functools import cache, cached_property, partial
from typing import Literal, overload

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.Pharm2D import Generate as Generate2D
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

from .base import Drawable
from .featurize import atom_features_simple, bond_features_simple, tokenize_smiles
from .gaussian_ph4 import GaussianPH4Generator
from .allinacp4_ph4 import FEATURE_TYPES, find_matches, aro_patterns as find_ARO, hbd_patterns as find_HBD, hba_patterns as find_HBA, pos_patterns as find_POS, neg_patterns as find_NEG, hyd_patterns as find_HYD

from e3fp.pipeline import fprints_from_mol
from e3fp.conformer.util import mol_from_sdf
from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint

@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class FingerprintOption:
    type: str = "morgan"
    # Morgan
    morgan_radius: int = 2
    morgan_n_bits: int = 256
    # RDKit
    rdkit_fp_size: int = 2048
    # Ph4
    ph4_dim: int = 840
    # Gaussian PH4 options
    gaussian_sigma: float = 1.0
    gaussian_distance_cutoff: float = 12.0
    gaussian_normalize: bool = True
    # E3FP options
    e3fp_bits: int = 4096
    e3fp_radius_multiplier: float = 1.5
    e3fp_max_energy_diff: float = 20.0
    e3fp_first: int = 3
    e3fp_rdkit_invariants: bool = True

    def __post_init__(self):
        supported_types = ("morgan", "rdkit", "gobbi_pharm2d", "ph4", "gaussian_ph4", "e3fp")
        if self.type not in supported_types:
            raise ValueError(f"Unsupported fingerprint type: {self.type}")

    @classmethod
    def morgan_for_tanimoto_similarity(cls):
        return FingerprintOption(
            type="morgan",
            morgan_radius=2,
            morgan_n_bits=4096,
        )

    @classmethod
    def gobbi_pharm2d(cls):
        return FingerprintOption(
            type="gobbi_pharm2d",
        )

    @classmethod
    def morgan_for_building_blocks(cls):
        return FingerprintOption(
            type="morgan",
            morgan_radius=2,
            morgan_n_bits=256,
        )

    @classmethod
    def rdkit(cls):
        return FingerprintOption(
            type="rdkit",
        )
    
    @classmethod
    def ph4(cls):
        return FingerprintOption(
            type="ph4",
            ph4_dim=840,
        )
    
    @classmethod
    def gaussian_ph4(cls):
        return FingerprintOption(
            type="gaussian_ph4",
            gaussian_sigma=1.0,
            gaussian_distance_cutoff=12.0,
            gaussian_normalize=True,
        )

    @classmethod
    def e3fp(cls):
        return FingerprintOption(
            type="e3fp",
            e3fp_bits=4096,
            e3fp_radius_multiplier=1.5,
            e3fp_max_energy_diff=20.0,
            e3fp_first=3,
            e3fp_rdkit_invariants=True,
        )

    @property
    def dim(self) -> int:
        if self.type == "morgan":
            return self.morgan_n_bits
        elif self.type == "rdkit":
            return self.rdkit_fp_size
        elif self.type == "gobbi_pharm2d":
            return 39972
        elif self.type == "ph4":
            return self.ph4_dim 
        elif self.type == "gaussian_ph4":
            return 63
        elif self.type == "e3fp":
            return self.e3fp_bits
        raise ValueError(f"Unsupported fingerprint type: {self.type}")


class Molecule(Drawable):
    def __init__(self, smiles: str) -> None:
        super().__init__()
        self._smiles = smiles.strip()

    @classmethod
    def from_rdmol(cls, rdmol: Chem.Mol) -> "Molecule":
        return cls(Chem.MolToSmiles(rdmol))

    def __getstate__(self):
        return self._smiles

    def __setstate__(self, state):
        self._smiles = state

    @property
    def smiles(self) -> str:
        return self._smiles

    @cached_property
    def _rdmol(self):
        return Chem.MolFromSmiles(self._smiles)

    @cached_property
    def _rdmol_no_hs(self):
        return Chem.RemoveHs(self._rdmol)

    @cached_property
    def is_valid(self) -> bool:
        return self._rdmol is not None

    @cached_property
    def csmiles(self) -> str:
        return Chem.MolToSmiles(self._rdmol, canonical=True, isomericSmiles=False)

    @cached_property
    def num_atoms(self) -> int:
        return self._rdmol.GetNumAtoms()

    def draw(self, size: int = 100, svg: bool = False):
        if svg:
            return Draw._moltoSVG(self._rdmol, sz=(size, size), highlights=[], legend=[], kekulize=True)
        else:
            return Draw.MolToImage(self._rdmol, size=(size, size), kekulize=True)

    def __hash__(self) -> int:
        return hash(self._smiles)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Molecule) and self.csmiles == __value.csmiles

    @cached_property
    def major_molecule(self) -> "Molecule":
        if "." in self.smiles:
            segs = self.smiles.split(".")
            segs.sort(key=lambda a: -len(a))
            return Molecule(segs[0])
        return self

    def featurize_simple(self) -> tuple[torch.Tensor, torch.Tensor]:
        mol = self._rdmol_no_hs
        atoms = mol.GetAtoms()

        atom_f = torch.zeros([len(atoms)], dtype=torch.long)
        bond_f = torch.zeros([len(atoms), len(atoms)], dtype=torch.long)

        for atom in atoms:
            idx = atom.GetIdx()
            atom_f[idx] = atom_features_simple(atom)
            for atom_j in atoms:
                jdx = atom_j.GetIdx()
                bond = mol.GetBondBetweenAtoms(idx, jdx)
                if bond is None:
                    continue
                bond_f[idx, jdx] = bond_features_simple(bond)

        return atom_f, bond_f

    def tokenize_csmiles(self) -> torch.Tensor:
        return torch.tensor(tokenize_smiles(self.csmiles), dtype=torch.long)

    @overload
    def get_fingerprint(self, option: FingerprintOption) -> np.ndarray: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[True]) -> Sequence[Literal[0, 1]]: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[False]) -> np.ndarray: ...

    def get_fingerprint(self, option: FingerprintOption, as_bitvec: bool = False):
        return self._get_fingerprint(option, as_bitvec)  # work-around for mypy check

    @cache
    def _get_fingerprint(self, option: FingerprintOption, as_bitvec: bool):
        if option.type == "morgan":
            bit_vec = AllChem.GetMorganFingerprintAsBitVect(self._rdmol, option.morgan_radius, option.morgan_n_bits)
        elif option.type == "rdkit":
            bit_vec = Chem.RDKFingerprint(self._rdmol, fpSize=option.rdkit_fp_size)
        elif option.type == "gobbi_pharm2d":
            bit_vec = DataStructs.cDataStructs.ConvertToExplicit(
                Generate2D.Gen2DFingerprint(self._rdmol, Gobbi_Pharm2D.factory)
            )
        elif option.type == "ph4":
            from .allinacp4_ph4 import compute_fingerprint_from_mol
            return compute_fingerprint_from_mol(self._rdmol)    
        elif option.type == "gaussian_ph4":
            gaussian_ph4_generator = GaussianPH4Generator(
                sigma=option.gaussian_sigma,
                distance_cutoff=option.gaussian_distance_cutoff,
                normalize=option.gaussian_normalize,
            )
            return gaussian_ph4_generator.get_fingerprint(self._rdmol)
        elif option.type == "e3fp":
            # Configure logging to suppress INFO messages
            import logging
            logging.getLogger("e3fp").setLevel(logging.WARNING)
            
            # Configure E3FP parameters
            fprint_params = {
                'bits': option.e3fp_bits,
                'radius_multiplier': option.e3fp_radius_multiplier,
                'rdkit_invariants': option.e3fp_rdkit_invariants
            }
            
            try:
                # Generate conformer if not already present
                if not self.has_conformer:
                    rdmol = Chem.AddHs(self._rdmol)
                    AllChem.EmbedMolecule(rdmol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(rdmol)
                    rdmol = Chem.RemoveHs(rdmol)
                    self.store_conformer(rdmol)
                
                # Use molecule with conformer
                fprints = fprints_from_mol(self._rdmol, fprint_params=fprint_params)
                
                # Convert to vector format
                if len(fprints) > 0:
                    # Convert to Fingerprint object first
                    fp = Fingerprint(fprints[0].indices, 
                                   bits=option.e3fp_bits,
                                   level=0,
                                   name=self.smiles)
                    
                    # Get sparse vector representation
                    vec = fp.to_vector(sparse=True)
                    
                    if as_bitvec:
                        return vec
                    # Convert to float32 while keeping sparsity
                    return vec.astype(np.float32)
                else:
                    # Return zero vector if no fingerprints generated
                    if as_bitvec:
                        return Fingerprint([], bits=option.e3fp_bits).to_vector(sparse=True)
                    return Fingerprint([], bits=option.e3fp_bits).to_vector(sparse=True).astype(np.float32)
                    
            except Exception as e:
                print(f"Warning: E3FP fingerprint generation failed: {str(e)}")
                if as_bitvec:
                    return Fingerprint([], bits=option.e3fp_bits).to_vector(sparse=True)
                return Fingerprint([], bits=option.e3fp_bits).to_vector(sparse=True).astype(np.float32)
        else:
            raise ValueError(f"Unsupported fingerprint type: {option.type}")

        if as_bitvec:
            return bit_vec
        feat = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bit_vec, feat)
        return feat

    @cached_property
    def scaffold(self) -> "Molecule":
        s = Molecule.from_rdmol(MurckoScaffold.GetScaffoldForMol(self._rdmol))
        if not s.is_valid:
            s = self
        return s

    def tanimoto_similarity(self, other: "Molecule", fp_option: FingerprintOption) -> float:
        fp1 = self.get_fingerprint(fp_option, as_bitvec=True)
        fp2 = other.get_fingerprint(fp_option, as_bitvec=True)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def tanimoto_similarity_counts(self, other: "Molecule", fp_option: FingerprintOption) -> float:
        if fp_option.type != "ph4":
            return self.tanimoto_similarity(other, fp_option)
        
        fp1 = self.get_fingerprint(fp_option, as_bitvec=False)
        fp2 = other.get_fingerprint(fp_option, as_bitvec=False)
        intersection = np.sum(np.minimum(fp1, fp2))
        union = np.sum(np.maximum(fp1, fp2))
        return float(intersection) / union if union > 0 else 0.0

    def dice_similarity(self, other: "Molecule", fp_option: FingerprintOption) -> float:
        fp1 = self.get_fingerprint(fp_option, as_bitvec=True)
        fp2 = other.get_fingerprint(fp_option, as_bitvec=True)
        return DataStructs.DiceSimilarity(fp1, fp2)

    @cache
    def sim(
        self,
        other: "Molecule",
        fp_option: FingerprintOption = FingerprintOption.morgan_for_tanimoto_similarity(),
    ) -> float:
        if fp_option.type == "ph4" or fp_option.type == "gaussian_ph4":
            return self.tanimoto_similarity_counts(other, fp_option)
        else:
            return self.tanimoto_similarity(other, fp_option)
        
    @cached_property
    def csmiles_md5(self) -> bytes:
        return hashlib.md5(self.csmiles.encode()).digest()

    @cached_property
    def csmiles_sha256(self) -> bytes:
        return hashlib.sha256(self.csmiles.encode()).digest()

    @property
    def has_conformer(self) -> bool:
        return self._rdmol.GetNumConformers() > 0

    def store_conformer(self, rdmol_with_conformer: Chem.Mol) -> None:
        """Store a conformer from another RDKit molecule"""
        if rdmol_with_conformer.GetNumConformers() == 0:
            raise ValueError("Input molecule has no conformers")
        
        # Copy the conformer to our molecule
        conformer = rdmol_with_conformer.GetConformer()
        self._rdmol.AddConformer(conformer)

    def get_pharmacophore_features(self, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get pharmacophore features and their 3D coordinates using SMARTS patterns from allinacp4_ph4"""
        if not self._rdmol.GetNumConformers():
            print(" ERROR No conformer found")
            #return torch.tensor([], device=device), torch.tensor([], device=device)
            raise ValueError(f"No conformer found for molecule: {self.smiles}")

            
        feature_patterns = {
            'ARO': find_ARO,
            'HBD': find_HBD,
            'HBA': find_HBA,
            'POS': find_POS,
            'NEG': find_NEG,
            'HYD': find_HYD
        }
        
        coords = []
        types = []
        
        for idx, (feat_type, patterns) in enumerate(feature_patterns.items()):
            positions = find_matches(self._rdmol, patterns)
            if positions:
                coords.extend(positions)
                types.extend([idx] * len(positions))
        
        if not coords:
            print("No pharmacophore features found")
            return torch.tensor([], device=device), torch.tensor([], device=device)
            
        # Convert to torch tensors
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        types_tensor = torch.tensor(types, dtype=torch.long, device=device)
            
        return coords_tensor, types_tensor
'''
    def get_pharmacophore_grid(self, grid_resolution=0.5, box_size=15) -> torch.Tensor:
        """Convert pharmacophore features to a grid representation"""
        coords, types = self.get_pharmacophore_features()
        if len(coords) == 0:
            return torch.zeros((grid_size, grid_size, grid_size, len(FEATURE_TYPES)))
        
        grid_size = int(2 * box_size / grid_resolution) + 1
        feature_grids = torch.zeros((grid_size, grid_size, grid_size, len(FEATURE_TYPES)), 
                                  device=coords.device)
        
        # Center of the grid
        center = torch.tensor([box_size, box_size, box_size], device=coords.device)
        
        # Fill grids with gaussian representations of features
        sigma = grid_resolution
        for i in range(len(coords)):
            pos = coords[i]
            feat_type = types[i]
            
            # Convert position to grid coordinates
            grid_pos = ((pos + center) / grid_resolution).long()
            
            # Create coordinate grids
            x = torch.arange(grid_size, device=coords.device)
            y = torch.arange(grid_size, device=coords.device)
            z = torch.arange(grid_size, device=coords.device)
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            
            # Compute gaussian
            gaussian = torch.exp(-((X - grid_pos[0])**2 + 
                                 (Y - grid_pos[1])**2 + 
                                 (Z - grid_pos[2])**2) / (2 * sigma**2))
            
            feature_grids[..., feat_type] += gaussian
        
        return feature_grids
'''

def read_mol_file(
    path: os.PathLike,
    major_only: bool = True,
    drop_duplicates: bool = True,
    show_pbar: bool = True,
    smiles_col: str | None = None,
    pbar_fn=partial(tqdm, desc="Reading"),
) -> Iterable[Molecule]:
    path = pathlib.Path(path)
    if path.suffix == ".sdf":
        #if fp_option and fp_option.type == "ph4":
        #    f = Chem.SDMolSupplier(str(path), removeHs=False)
        #else:
        f = Chem.SDMolSupplier(str(path))
    elif path.suffix == ".smi":
        f = Chem.SmilesMolSupplier(str(path))
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        if smiles_col is None:
            if "smiles" in df.columns:
                smiles_col = "smiles"
            elif "SMILES" in df.columns:
                smiles_col = "SMILES"
            else:
                raise ValueError(f"Cannot find SMILES column in {path}")
        f = (Chem.MolFromSmiles(smiles) for smiles in df[smiles_col])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    visited: set[str] = set()
    if show_pbar:
        f_iter = pbar_fn(f)
    else:
        f_iter = f
    for rdmol in f_iter:
        if rdmol is not None:
            mol = Molecule.from_rdmol(rdmol)
            if major_only:
                mol = mol.major_molecule
            if drop_duplicates and mol.csmiles in visited:
                continue
            yield mol
            visited.add(mol.csmiles)


def write_to_smi(path: os.PathLike, mols: Sequence[Molecule]):
    with open(path, "w") as f:
        for mol in mols:
            f.write(f"{mol.smiles}\n")
