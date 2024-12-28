#!/bin/bash
#SBATCH --output=/itet-stor/sdivita/net_scratch/originale/ChemProjector/jobs/%j.out
#SBATCH --error=/itet-stor/sdivita/net_scratch/originale/ChemProjector/jobs/%j.err
#SBATCH --mem=600G  # Increased from 32G to 128G for large molecule processing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00  # Added 48-hour time limit since this is a long job

ETH_USERNAME=sdivita
PROJECT_NAME=ChemProjector
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/originale/${PROJECT_NAME}
CONDA_ENVIRONMENT=chemprojector
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Set up temporary directory with more space
TMPDIR="/itet-stor/${ETH_USERNAME}/net_scratch/tmp"
mkdir -p "${TMPDIR}"
export TMPDIR

# Log information
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# Activate conda
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"

# Change to project directory
cd ${DIRECTORY}

# Execute shape creation script
python scripts/validation/create_just_shapes.py

# Log completion
echo "Finished at: $(date)"

# Exit successfully
exit 0 