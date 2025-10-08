#!/bin/bash
#SBATCH --job-name=comp_null_ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=amilan
#SBATCH --output=logs/comp_null_ext_%j.out
#SBATCH --error=logs/comp_null_ext_%j.err
#SBATCH --qos=normal

# =====================================
# Extended Time Compositional Null Script
# =====================================
# This is a fallback script that uses the original notebook
# but with extended time limit (12 hours) to ensure completion
# =====================================

echo "====================================="
echo "Starting Compositional Null (Extended Time)"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Time limit: 12 hours"
echo "Start time: $(date)"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directories
mkdir -p results/compositional_null
mkdir -p logs

# Run original notebook with extended time
echo "Running notebook 14 with extended time limit..."

papermill notebooks/14_fast_compositional_null.ipynb \
    notebooks/executed/14_fast_compositional_null_executed.ipynb \
    --log-output \
    --progress-bar

echo "End time: $(date)"
echo "====================================="