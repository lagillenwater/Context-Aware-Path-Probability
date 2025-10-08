#!/bin/bash
#SBATCH --job-name=nb5_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --output=logs/nb5_fix_%j.out
#SBATCH --error=logs/nb5_fix_%j.err
#SBATCH --qos=normal

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP


# Create output directories
mkdir -p logs
mkdir -p notebooks/executed

# Run fixed notebook 5
echo "===== Running Fixed Notebook 5 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

papermill notebooks/5_model_testing_summary.ipynb \
    notebooks/executed/5_model_testing_summary_executed.ipynb \
    --log-output \
    --progress-bar

echo "End time: $(date)"
echo "===== Notebook 5 Complete ====="
