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
@click.option(
    "--out",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/processed/all/fpindex_pharmacomit.pkl")
)
@click.option(
    "--batch-size",
    type=int,
    default=20000,
    help="Number of molecules to process in each batch"
)
def fpindex(
    model_config: DictConfig,
    molecule: pathlib.Path,
    out: pathlib.Path,
    batch_size: int,
):
    out.parent.mkdir(parents=True, exist_ok=True)
    fp_option = FingerprintOption(**model_config.chem.fp_option)
    
    try:
        print("\nStarting fingerprint index creation...")
        print(f"Input file: {molecule}")
        print(f"Output file: {out}")
        print(f"Batch size: {batch_size}")
        
        fpindex = create_fingerprint_index_cache(
            molecule_path=molecule,
            cache_path=out,
            fp_option=fp_option,
            batch_size=batch_size
        )
        
        print("\nProcess completed successfully!")
        print(f"Number of molecules processed: {len(fpindex.molecules)}")
        print(f"FingerprintIndex saved to: {out}")
        
    except Exception as e:
        print(f"\nERROR: Process failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    fpindex()
