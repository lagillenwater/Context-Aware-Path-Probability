#!/bin/bash
#SBATCH --job-name=enhanced_model_testing
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --array=1-24
#SBATCH --output=../logs/enhanced_model_%A_%a.out
#SBATCH --error=../logs/enhanced_model_%A_%a.err

# Enhanced Model Testing with Degree Analysis HPC Script
#
# This script runs enhanced model testing with integrated degree-based analysis
# for comprehensive error analysis across all edge types.
#
# Usage:
#   sbatch scripts/enhanced_model_testing_hpc.sh

# Setup
module purge
module load python/3.8.5
module load gcc/10.3.0

# Activate environment
source ~/miniconda3/envs/hetionet/bin/activate

# Navigate to repository
cd $SLURM_SUBMIT_DIR/..

# Create output directories
mkdir -p results/enhanced_model_testing
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
echo "Enhanced model testing for: $EDGE_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../src:$PYTHONPATH"

# Run enhanced model testing with papermill
papermill \
    notebooks/5_model_testing_summary_with_degree_analysis.ipynb \
    results/enhanced_model_testing/${EDGE_TYPE}_enhanced_model_testing.ipynb \
    -p edge_types "['${EDGE_TYPE}']" \
    -p small_graph_mode False

if [ $? -eq 0 ]; then
    echo "✓ Enhanced model testing completed for $EDGE_TYPE"
else
    echo "✗ Enhanced model testing failed for $EDGE_TYPE"
    exit 1
fi

echo "Job completed at: $(date)"