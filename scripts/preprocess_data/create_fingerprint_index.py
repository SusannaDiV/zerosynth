import pathlib

import click
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem

from chemprojector.chem.fpindex import create_fingerprint_index_cache
from chemprojector.chem.mol import FingerprintOption

_default_sdf_path = pathlib.Path("/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf")


@click.command()
@click.option(
    "--model-config",
    type=OmegaConf.load,
    required=True,
)
@click.option(
    "--molecule",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_sdf_path,
)
@click.option("--out", type=click.Path(path_type=pathlib.Path), default=pathlib.Path("data/processed/all/fpindex_ph4_new.pkl"))
def fpindex(model_config: DictConfig, molecule: pathlib.Path, out: pathlib.Path):
    if out.exists():
        click.confirm(f"{out} already exists. Overwrite?", abort=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("\nDEBUG: Testing molecule suppliers")
    print("Testing SDMolSupplier:")
    mol = next(iter(Chem.SDMolSupplier(str(molecule), removeHs=False)))
    conf = mol.GetConformer()
    print(f"Is 3D: {conf.Is3D()}")
    print(f"First atom coords: {conf.GetAtomPosition(0).x}, {conf.GetAtomPosition(0).y}, {conf.GetAtomPosition(0).z}")

    print("\nTesting ForwardSDMolSupplier:")
    mol = next(iter(Chem.ForwardSDMolSupplier(str(molecule), removeHs=False)))
    conf = mol.GetConformer()
    print(f"Is 3D: {conf.Is3D()}")
    print(f"First atom coords: {conf.GetAtomPosition(0).x}, {conf.GetAtomPosition(0).y}, {conf.GetAtomPosition(0).z}")

    print("Config fp_option:", model_config.chem.fp_option)
    print("Config fp type:", model_config.chem.fp_option.type)
    print("Config fp type type:", type(model_config.chem.fp_option.type))
    
    fp_option = FingerprintOption(**model_config.chem.fp_option)
    print("ok")
    fpindex = create_fingerprint_index_cache(
        molecule_path=molecule,
        cache_path=out,
        fp_option=fp_option,
    )
    print(f"Number of molecules: {len(fpindex.molecules)}")
    print(f"Saved index to {out}")


if __name__ == "__main__":
    fpindex()
