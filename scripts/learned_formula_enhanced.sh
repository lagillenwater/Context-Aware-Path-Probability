#!/bin/bash
#SBATCH --job-name=learned_formula_enhanced
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --array=1-24
#SBATCH --output=../logs/learned_formula_%A_%a.out
#SBATCH --error=../logs/learned_formula_%A_%a.err

# Enhanced Learned Analytical Formula HPC Script
#
# This script runs enhanced learned analytical formula analysis with degree-based
# error decomposition across all edge types and formula variants.
#
# Usage:
#   sbatch scripts/learned_formula_enhanced_hpc.sh

# Setup
module purge
module load python/3.8.5
module load gcc/10.3.0

# Activate environment
source ~/miniconda3/envs/hetionet/bin/activate

# Navigate to repository
cd $SLURM_SUBMIT_DIR/..

# Create output directories
mkdir -p results/learned_analytical_enhanced_hpc
mkdir -p logs

# Define edge types array
EDGE_TYPES=(
    "AdG" "AeG" "AuG" "CbG" "CcSE" "CdG" "CpD" "CrC" "CtD" "CuG"
    "DaG" "DdG" "DlA" "DpS" "DrD" "DuG" "GcG" "GiG" "GpBP" "GpCC"
    "GpMF" "GpPW" "Gr>G" "PCiC"
)

# Get edge type for this array job
EDGE_TYPE=${EDGE_TYPES[$SLURM_ARRAY_TASK_ID-1]}

echo "=========================================="
echo "Enhanced learned formula for: $EDGE_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../src:$PYTHONPATH"

# Run analysis for each formula type
FORMULA_TYPES=("original" "extended" "polynomial")

for FORMULA_TYPE in "${FORMULA_TYPES[@]}"; do
    echo "Running $FORMULA_TYPE formula analysis for $EDGE_TYPE..."

    # Create output subdirectory
    OUTPUT_SUBDIR="results/learned_analytical_enhanced_hpc/${EDGE_TYPE}_${FORMULA_TYPE}"
    mkdir -p $OUTPUT_SUBDIR

    # Run enhanced learned formula analysis
    papermill \
        notebooks/8_learned_analytical_formula_with_degree_analysis.ipynb \
        ${OUTPUT_SUBDIR}/${EDGE_TYPE}_${FORMULA_TYPE}_enhanced_analysis.ipynb \
        -p edge_type "$EDGE_TYPE" \
        -p formula_type "$FORMULA_TYPE" \
        -p small_graph_mode False \
        -p N_candidates "[2, 3, 5, 7, 10, 15, 20, 30, 40, 50]" \
        -p convergence_threshold 0.02 \
        -p target_metric "correlation" \
        -p min_metric_value 0.95

    if [ $? -eq 0 ]; then
        echo "✓ $FORMULA_TYPE analysis completed for $EDGE_TYPE"
    else
        echo "✗ $FORMULA_TYPE analysis failed for $EDGE_TYPE"
        # Continue with other formula types even if one fails
    fi
done

echo "Job completed at: $(date)"