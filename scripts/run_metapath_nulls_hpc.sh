#!/bin/bash
#SBATCH --job-name=metapath_nulls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --output=logs/metapath_nulls_%j.out
#SBATCH --error=logs/metapath_nulls_%j.err
#SBATCH --qos=normal

# Load modules
module load python/3.9
module load anaconda

# Activate conda environment
source activate hetionet_env

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directories
mkdir -p results/metapath_nulls
mkdir -p logs

# Run notebook 15: Metapath null distributions
echo "===== Starting Metapath Null Generation ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

papermill notebooks/15_metapath_null_distributions.ipynb \
    notebooks/executed/15_metapath_null_distributions_executed.ipynb \
    --log-output \
    --progress-bar

echo "End time: $(date)"
echo "===== Metapath Null Generation Complete ====="
