#!/bin/bash
#SBATCH --job-name=null_train
#SBATCH --array=1-24                # 24 edge types in parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # Sufficient for single edge type
#SBATCH --mem=8GB                   # Sufficient for any edge type (peak ~5GB)
#SBATCH --time=01:30:00             # Longest edge type ~1 hour
#SBATCH --partition=amilan
#SBATCH --output=logs/model_training/null_models/null_training_%A_%a.out
#SBATCH --error=logs/model_training/null_models/null_training_%A_%a.err
#SBATCH --qos=normal

# Array of edge types (alphabetical order)
EDGE_TYPES=(
    "AdG" "AeG" "AuG" "CbG" "CcSE" "CdG" "CpD" "CrC" "CtD" "CuG"
    "DaG" "DdG" "DlA" "DpS" "DrD" "DuG" "GcG" "GiG" "GpBP" "GpCC"
    "GpMF" "GpPW" "Gr>G" "PCiC"
)

# Get edge type for this array task
EDGE_TYPE=${EDGE_TYPES[$((SLURM_ARRAY_TASK_ID-1))]}

echo "===== Null Model Training: ${EDGE_TYPE} ====="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/24"
echo "Edge type: ${EDGE_TYPE}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 8GB (peak usage ~1-5GB depending on edge type)"
echo "Training: Individual binary samples from 20 permutations"
echo "Expected: r > 0.85 vs TRUE empirical frequencies"
echo "Start time: $(date)"
echo ""

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Set environment variables for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directories
mkdir -p results/null_models
mkdir -p logs/model_training/null_models
mkdir -p notebooks/executed

# Run notebook 13 for this specific edge type
papermill notebooks/13_null_model_training.ipynb \
    notebooks/executed/13_null_model_training_${EDGE_TYPE}_executed.ipynb \
    -p edge_types_to_process "['${EDGE_TYPE}']" \
    -p training_perm_range "(1, 21)" \
    -p validation_perm_range "(21, 31)" \
    --log-output \
    --progress-bar

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ ${EDGE_TYPE} training completed successfully"
else
    echo "✗ ${EDGE_TYPE} training failed with exit code ${EXIT_CODE}"
fi

echo "===== ${EDGE_TYPE} Complete ====="

exit $EXIT_CODE
