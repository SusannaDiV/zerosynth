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
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from existing intermediate pickles"
)
def fpindex(
    model_config: DictConfig,
    molecule: pathlib.Path,
    out: pathlib.Path,
    batch_size: int,
    resume: bool,
):
    # Check intermediate directory
    intermediate_dir = out.parent / "intermediate_pickles"
    if intermediate_dir.exists() and not resume:
        click.confirm(
            "Intermediate pickles found. Delete and start fresh?", 
            abort=True
        )
        import shutil
        shutil.rmtree(intermediate_dir)
    
    out.parent.mkdir(parents=True, exist_ok=True)
    fp_option = FingerprintOption(**model_config.chem.fp_option)
    
    try:
        print("\nStarting fingerprint index creation...")
        print(f"Input file: {molecule}")
        print(f"Output file: {out}")
        print(f"Batch size: {batch_size}")
        print(f"Resume mode: {resume}")
        
        fpindex = create_fingerprint_index_cache(
            molecule_path=molecule,
            cache_path=out,
            fp_option=fp_option,
            batch_size=batch_size,
            resume=resume
        )
        
        print("\nProcess completed successfully!")
        print(f"Number of molecules processed: {len(fpindex.molecules)}")
        print(f"Main index saved to: {out}")
        print(f"Shape patches saved to: {out.parent / (out.stem + '_shape_patches.pkl')}")
        print(f"Ph4 patches saved to: {out.parent / (out.stem + '_ph4_patches.pkl')}")
        
    except Exception as e:
        print(f"\nERROR: Process failed with error: {str(e)}")
        print("\nIntermediate files are preserved in:")
        print(f"- {intermediate_dir}")
        print("You can resume processing using --resume flag")
        raise

if __name__ == "__main__":
    fpindex()
