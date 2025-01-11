import pathlib

import click
from omegaconf import DictConfig, OmegaConf

from chemprojector.chem.fpindex import create_fingerprint_index_cache
from chemprojector.chem.mol import FingerprintOption

_default_sdf_path = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_249948cmpd_20241108.sdf")


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
@click.option("--out", type=click.Path(path_type=pathlib.Path), default=pathlib.Path("data/processed/all/fpindex_pharmacomit.pkl"))
@click.option("--num-molecules", type=int, default=100, help="Number of molecules to process")
def fpindex(model_config: DictConfig, molecule: pathlib.Path, out: pathlib.Path, num_molecules: int):
    if out.exists():
        click.confirm(f"{out} already exists. Overwrite?", abort=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    fp_option = FingerprintOption(**model_config.chem.fp_option)
    
    # Add limit to number of molecules
    print(f"\nProcessing first {num_molecules} molecules...")
    fpindex = create_fingerprint_index_cache(
        molecule_path=molecule,
        cache_path=out,
        fp_option=fp_option,
        max_molecules=num_molecules  # New parameter
    )
    print(f"Number of molecules processed: {len(fpindex.molecules)}")
    print(f"Saved index to {out}")


if __name__ == "__main__":
    fpindex()
