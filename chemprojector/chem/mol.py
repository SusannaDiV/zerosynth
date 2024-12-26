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


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class FingerprintOption:
    type: str = "ph4"
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

    def __post_init__(self):
        supported_types = ("ph4", "gaussian_ph4", "morgan", "rdkit", "gobbi_pharm2d")
        if self.type not in supported_types:
            raise ValueError(f"Unsupported fingerprint type: {self.type}")

    @classmethod
    def ph4(cls):
        return FingerprintOption(
            type="ph4",
            ph4_dim=840,
        )
    
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

    @property
    def dim(self) -> int:
        if self.type == "ph4":
            return self.ph4_dim
        if self.type == "morgan":
            return self.morgan_n_bits
        elif self.type == "rdkit":
            return self.rdkit_fp_size
        elif self.type == "gobbi_pharm2d":
            return 39972
        raise ValueError(f"Unsupported fingerprint type: {self.type}")


class Molecule(Drawable):
    def __init__(self, smiles: str = None, rdmol = None):
        """Initialize Molecule with either SMILES or RDKit Mol object"""
        if rdmol is not None:
            self._rdmol = rdmol
        elif smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            self._rdmol = mol
        else:
            raise ValueError("Either smiles or rdmol must be provided")

    @property
    def smiles(self) -> str:
        return self._smiles

    @classmethod
    def from_rdmol(cls, rdmol: Chem.Mol) -> "Molecule":
        smiles = Chem.MolToSmiles(rdmol, canonical=True)
        return cls(smiles, rdmol=rdmol)

    @classmethod
    def from_smiles(cls, smiles: str) -> 'Molecule':
        """Create a Molecule instance from SMILES string"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Generate 3D conformer
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Create Molecule instance
        return cls(rdmol=mol)

    @cached_property
    def _rdmol(self):
        if self._original_rdmol is not None:
            return self._original_rdmol
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
        """Generate simple atom and bond features"""
        # Basic atom features
        atom_features = []
        for atom in self._rdmol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic())
            ]
            atom_features.append(features)
            
        # Basic bond features
        bond_features = []
        for bond in self._rdmol.GetBonds():
            features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing())
            ]
            bond_features.append(features)
            
        return atom_features, bond_features

    def tokenize_csmiles(self) -> torch.Tensor:
        return torch.tensor(tokenize_smiles(self.csmiles), dtype=torch.long)

    @overload
    def get_fingerprint(self, option: FingerprintOption) -> np.ndarray: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[True]) -> Sequence[Literal[0, 1]]: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[False]) -> np.ndarray: ...

    @cache
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: bool = False):
        if option.type == "ph4":
            from .allinacp4_ph4 import compute_fingerprint_from_mol
            return compute_fingerprint_from_mol(self._rdmol).flatten()
        elif option.type == "morgan":
            bit_vec = AllChem.GetMorganFingerprintAsBitVect(self._rdmol, option.morgan_radius, option.morgan_n_bits)
        elif option.type == "rdkit":
            bit_vec = Chem.RDKFingerprint(self._rdmol, fpSize=option.rdkit_fp_size)
        elif option.type == "gobbi_pharm2d":
            bit_vec = DataStructs.cDataStructs.ConvertToExplicit(
                Generate2D.Gen2DFingerprint(self._rdmol, Gobbi_Pharm2D.factory)
            )
        elif option.type == "gaussian_ph4":
            generator = GaussianPH4Generator(
                sigma=option.gaussian_sigma,
                distance_cutoff=option.gaussian_distance_cutoff,
                normalize=option.gaussian_normalize
            )
            return generator.get_fingerprint(self._rdmol)
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

    def dice_similarity(self, other: "Molecule", fp_option: FingerprintOption) -> float:
        fp1 = self.get_fingerprint(fp_option, as_bitvec=True)
        fp2 = other.get_fingerprint(fp_option, as_bitvec=True)
        return DataStructs.DiceSimilarity(fp1, fp2)

    def manhattan_similarity(self, other: "Molecule", fp_option: FingerprintOption) -> float:
        """Manhattan similarity for ph4 fingerprints"""
        if fp_option.type != "ph4":
            raise ValueError("Manhattan similarity is only supported for ph4 fingerprints")
        
        fp1 = self.get_fingerprint(fp_option, as_bitvec=False)
        fp2 = other.get_fingerprint(fp_option, as_bitvec=False)
        manhattan_dist = np.sum(np.abs(fp1 - fp2))
        max_possible_dist = np.sum(np.maximum(fp1, fp2))  # Normalize by maximum possible distance
        return 1 - (manhattan_dist / max_possible_dist if max_possible_dist > 0 else 0)

    @cache
    def sim(
        self,
        other: "Molecule",
        fp_option: FingerprintOption = FingerprintOption.morgan_for_tanimoto_similarity(),
    ) -> float:
        if fp_option.type == "ph4":
            return self.manhattan_similarity(other, fp_option)
        else:
            return self.tanimoto_similarity(other, fp_option)

    @cached_property
    def csmiles_md5(self) -> bytes:
        return hashlib.md5(self.csmiles.encode()).digest()

    @cached_property
    def csmiles_sha256(self) -> bytes:
        return hashlib.sha256(self.csmiles.encode()).digest()


def read_mol_file(
    path: os.PathLike,
    major_only: bool = True,
    drop_duplicates: bool = True,
    show_pbar: bool = True,
    smiles_col: str | None = None,
    pbar_fn=partial(tqdm, desc="Reading"),
    fp_option: FingerprintOption | None = None,
) -> Iterable[Molecule]:
    path = pathlib.Path(path)
    if path.suffix == ".sdf":
        if fp_option and fp_option.type == "ph4":
            f = Chem.SDMolSupplier(str(path), removeHs=False)
        else:
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
