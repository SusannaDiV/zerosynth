import pathlib
import click
from chemprojector.chem.fpindex import create_fingerprint_index_cache
from chemprojector.chem.mol import FingerprintOption

@click.command()
@click.option(
    "--sdf-file",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/processed/all/fpindex_ph4.pkl"),
)
def main(sdf_file: pathlib.Path, output: pathlib.Path):
    if output.exists():
        click.confirm(f"{output} already exists. Overwrite?", abort=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Configure ph4 fingerprint options
    fp_option = FingerprintOption(
        type="ph4",
        ph4_temp_dir=pathlib.Path("data/temp/ph4")  # Add this line
    )

    # Create fingerprint index
    fpindex = create_fingerprint_index_cache(
        molecule_path=sdf_file,
        cache_path=output,
        fp_option=fp_option,
    )
    
    print(f"Processed {len(fpindex.molecules)} molecules")
    print(f"Saved index to {output}")

if __name__ == "__main__":
    main() 