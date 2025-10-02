#!/bin/bash

#SBATCH --job-name=min_perm_ml
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_min_perm_ml_%a.log
#SBATCH --error=../logs/error_min_perm_ml_%a.log
#SBATCH --time=08:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --array=1-24

# Minimum Permutations Analysis for ML Models
# This script runs the minimum permutations analysis (notebook 6) for all edge types
# Tests how many permutations ML models need to learn empirical frequencies

# Exit if any command fails
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Get the directory of this script and define base paths relative to it
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"
logs_dir="${BASE_DIR}/logs"
results_dir="${BASE_DIR}/results/minimum_permutations_ml"

# Create logs directory if it doesn't exist
mkdir -p $logs_dir
mkdir -p $results_dir

echo "****** Running minimum permutations analysis for job array index $SLURM_ARRAY_TASK_ID ******"
echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Define array of edge files
EDGE_FILES=(
    "AdG.sparse.npz"
    "AeG.sparse.npz"
    "AuG.sparse.npz"
    "CbG.sparse.npz"
    "CcSE.sparse.npz"
    "CdG.sparse.npz"
    "CpD.sparse.npz"
    "CrC.sparse.npz"
    "CtD.sparse.npz"
    "CuG.sparse.npz"
    "DaG.sparse.npz"
    "DdG.sparse.npz"
    "DlA.sparse.npz"
    "DpS.sparse.npz"
    "DrD.sparse.npz"
    "DuG.sparse.npz"
    "GcG.sparse.npz"
    "GiG.sparse.npz"
    "GpBP.sparse.npz"
    "GpCC.sparse.npz"
    "GpMF.sparse.npz"
    "GpPW.sparse.npz"
    "Gr>G.sparse.npz"
    "PCiC.sparse.npz"
)

# Get the edge file for this array task
EDGE_FILE=${EDGE_FILES[$((SLURM_ARRAY_TASK_ID-1))]}
EDGE_TYPE=${EDGE_FILE%.sparse.npz}

echo "Processing edge type: $EDGE_TYPE"
echo "Edge file: $EDGE_FILE"

# Check if edge type has empirical frequencies
EMPIRICAL_FILE="${BASE_DIR}/results/empirical_edge_frequencies/edge_frequency_by_degree_${EDGE_TYPE}.csv"
if [ ! -f "$EMPIRICAL_FILE" ]; then
    echo "ERROR: Empirical frequency file not found: $EMPIRICAL_FILE"
    echo "Run notebook 3 (edge_frequency_by_degree) first"
    exit 1
fi

echo "Empirical frequency file found: $EMPIRICAL_FILE"

# Define paths
input_notebook="${notebooks_path}/6_minimum_permutations_analysis.ipynb"
output_notebook="${notebooks_path}/outputs/minimum_permutations_ml/6_min_perm_ml_${EDGE_TYPE}.ipynb"

echo "Input notebook: $input_notebook"
echo "Output notebook: $output_notebook"

# Create output directory if it doesn't exist
mkdir -p "${notebooks_path}/outputs/minimum_permutations_ml"
echo "Created output directory: ${notebooks_path}/outputs/minimum_permutations_ml"

# Papermill parameters
MAX_PERMUTATIONS="50"
CONVERGENCE_THRESHOLD="0.02"
TARGET_METRIC="correlation"
MIN_METRIC_VALUE="0.90"
RANDOM_SEED="42"

echo ""
echo "Parameters:"
echo "  Edge type: $EDGE_TYPE"
echo "  Max permutations: $MAX_PERMUTATIONS"
echo "  Convergence threshold: $CONVERGENCE_THRESHOLD"
echo "  Target metric: $TARGET_METRIC >= $MIN_METRIC_VALUE"
echo "  Random seed: $RANDOM_SEED"
echo ""

# Run papermill with parameters
echo "Starting papermill execution for edge type: $EDGE_TYPE"
echo "------------------------------------------------------------------------"

papermill "$input_notebook" "$output_notebook" \
    -p edge_type "$EDGE_TYPE" \
    -p max_permutations "$MAX_PERMUTATIONS" \
    -p convergence_threshold "$CONVERGENCE_THRESHOLD" \
    -p target_metric "$TARGET_METRIC" \
    -p min_metric_value "$MIN_METRIC_VALUE" \
    -p random_seed "$RANDOM_SEED" \
    --log-output \
    --progress-bar

PAPERMILL_EXIT_CODE=$?

echo ""
echo "------------------------------------------------------------------------"
if [ $PAPERMILL_EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Notebook executed successfully"
    echo "  Output: $output_notebook"

    # Check if results were created
    RESULTS_FILE="${results_dir}/${EDGE_TYPE}_results/${EDGE_TYPE}_summary.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "  Results: $RESULTS_FILE"

        # Extract N_min values for quick reference
        if command -v jq &> /dev/null; then
            echo "  N_min values:"
            jq -r '.models | to_entries[] | "    \(.key): \(.value.N_min)"' "$RESULTS_FILE"
        fi
    fi
else
    echo "✗ ERROR: Notebook execution failed with exit code $PAPERMILL_EXIT_CODE"
    echo "  Check logs: $logs_dir/error_min_perm_ml_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
fi

echo ""
echo "========================================================================"
echo "End time: $(date)"
echo "========================================================================"

exit $PAPERMILL_EXIT_CODE
