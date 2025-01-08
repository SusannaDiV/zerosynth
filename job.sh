#!/bin/bash
#SBATCH --output=/itet-stor/sdivita/net_scratch/shitong/ChemProjector/jobs/%j.out
#SBATCH --error=/itet-stor/sdivita/net_scratch/shitong/ChemProjector/jobs/%j.err
##SBATCH --mem=98G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --nodelist=tikgpu[08-10]  # For Titan RTX GPUs
ETH_USERNAME=sdivita
PROJECT_NAME=ChemProjector
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/shitong/${PROJECT_NAME}
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
python train.py configs/shape_default.yml --batch-size 16 --num-workers 4 --devices 4 --log-dir /itet-stor/sdivita/net_scratch/shitong/ChemProjector/runs  #&& python scripts/sbdd/10-run_docking.py && python scripts/sbdd/20-summarize.py

# Log completion
echo "Finished at: $(date)"

# Exit successfully
exit 0 