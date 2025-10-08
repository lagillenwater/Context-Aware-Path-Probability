#!/bin/bash
#SBATCH --job-name=comp_null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --output=logs/comp_null_%j.out
#SBATCH --error=logs/comp_null_%j.err
#SBATCH --qos=normal

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directories
mkdir -p results/compositional_null
mkdir -p logs

# Run notebook 14: Fast compositional null
echo "===== Starting Compositional Null Calculation ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

papermill notebooks/14_fast_compositional_null.ipynb \
    notebooks/executed/14_fast_compositional_null_executed.ipynb \
    --log-output \
    --progress-bar

echo "End time: $(date)"
echo "===== Compositional Null Complete ====="
