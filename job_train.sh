#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/sdivita/net_scratch/originale/ChemProjector/jobs/%j.out
#SBATCH --error=/itet-stor/sdivita/net_scratch/originale/ChemProjector/jobs/%j.err
#SBATCH --mem=664G  # Memory for training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4  # Request 1 GPU
#SBATCH --time=24:00:00
##SBATCH --constraint='titan_rtx|tesla_v100|titan_xp'  # Request specific GPU types

ETH_USERNAME=sdivita
PROJECT_NAME=ChemProjector
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/originale/${PROJECT_NAME}
CONDA_ENVIRONMENT=chemprojector
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Set up temporary directory
TMPDIR="/itet-stor/${ETH_USERNAME}/net_scratch/tmp"
mkdir -p "${TMPDIR}"
export TMPDIR

# Log information
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "GPU requested: $CUDA_VISIBLE_DEVICES"

# Activate conda
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"

# Change to project directory
cd ${DIRECTORY}

# Execute training script with optimized parameters
python train.py configs/original_default.yml \
    --batch-size 32 \
    --num-workers 4 \
    --devices 1 \
    --log-dir runs/original_training

# Log completion
echo "Finished at: $(date)"

# Exit successfully
exit 0 