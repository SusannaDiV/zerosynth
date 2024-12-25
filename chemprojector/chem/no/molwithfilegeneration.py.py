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


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class FingerprintOption:
    type: str = "morgan"
    # Morgan
    morgan_radius: int = 2
    morgan_n_bits: int = 256
    # RDKit
    rdkit_fp_size: int = 2048
    # Ph4
    ph4_temp_dir: pathlib.Path | str = pathlib.Path("data/temp/ph4")
    dim: int = 840  # Default for Ph4: (6 * 7/2) feature pairs * 40 distance bins

    def __post_init__(self):
        supported_types = ("ph4", "morgan", "rdkit", "gobbi_pharm2d")
        if isinstance(self.ph4_temp_dir, str):
            object.__setattr__(self, 'ph4_temp_dir', pathlib.Path(self.ph4_temp_dir))
            
        if self.type == "ph4":
            # Ensure the temp directory exists
            self.ph4_temp_dir.mkdir(parents=True, exist_ok=True)
        if self.type not in supported_types:
            raise ValueError(f"Unsupported fingerprint type: {self.type}")

    @classmethod
    def ph4(cls):
        return FingerprintOption(
            type="ph4",
            dim=840,  # (6 * 7/2) feature pairs * 40 distance bins
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
            from .ph4_fp import Ph4Fingerprint
            return Ph4Fingerprint.dim()
        if self.type == "morgan":
            return self.morgan_n_bits
        elif self.type == "rdkit":
            return self.rdkit_fp_size
        elif self.type == "gobbi_pharm2d":
            return 39972
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

    def _generate_ph4(self, ph4_path: pathlib.Path) -> None:
    # Save molecule to temporary SDF file
        temp_sdf = ph4_path.with_suffix('.sdf')
        writer = Chem.SDWriter(str(temp_sdf))
        writer.write(self._rdmol)
        writer.close()

        # Run ACP4 to generate ph4 file
        from subprocess import run
        run(['/itet-stor/sdivita/net_scratch/originale/ChemProjector/third_party/ACP4/bin/acp4_ph4.py', '-i', str(temp_sdf), '-o', str(ph4_path)], check=True)
        
        # Clean up temporary SDF file
        temp_sdf.unlink()
    
    @cache
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: bool = False) -> np.ndarray:
        if option.type == "ph4":
            # Generate unique filename based on molecule hash
            ph4_path = option.ph4_temp_dir / f"{self.csmiles_md5.hex()}.ph4"
            
            # Generate ph4 file if it doesn't exist
            if not ph4_path.exists():
                self._generate_ph4(ph4_path)
                
            # Import the correct Ph4Fingerprint implementation
            from .allinacp4_ph4 import compute_fingerprint_from_mol
            # Compute fingerprint directly from molecule
            fp = compute_fingerprint_from_mol(self._rdmol)
            
            if as_bitvec:
                return (fp > 0).astype(np.uint8)
            return fp
        
        # Rest of the existing fingerprint code...
        if option.type == "morgan":
            bit_vec = AllChem.GetMorganFingerprintAsBitVect(self._rdmol, option.morgan_radius, option.morgan_n_bits)
        elif option.type == "rdkit":
            bit_vec = Chem.RDKFingerprint(self._rdmol, fpSize=option.rdkit_fp_size)
        elif option.type == "gobbi_pharm2d":
            bit_vec = DataStructs.cDataStructs.ConvertToExplicit(
                Generate2D.Gen2DFingerprint(self._rdmol, Gobbi_Pharm2D.factory)
            )
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
) -> Iterable[Molecule]:
    path = pathlib.Path(path)
    if path.suffix == ".sdf":
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
