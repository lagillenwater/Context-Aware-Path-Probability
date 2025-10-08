#!/bin/bash
#SBATCH --job-name=null_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --output=logs/null_training_%j.out
#SBATCH --error=logs/null_training_%j.err
#SBATCH --qos=normal

# Load modules
module load python/3.9
module load anaconda

# Activate conda environment
source activate hetionet_env

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directories
mkdir -p results/null_models
mkdir -p logs

# Run notebook 13: Train null models
echo "===== Starting Null Model Training ====="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "Start time: $(date)"

papermill notebooks/13_null_model_training.ipynb \
    notebooks/executed/13_null_model_training_executed.ipynb \
    --log-output \
    --progress-bar

echo "End time: $(date)"
echo "===== Null Model Training Complete ====="
