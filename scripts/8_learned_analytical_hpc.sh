#!/bin/bash
#SBATCH --job-name=learned_analytical
#SBATCH --output=logs/learned_analytical_%A_%a.out
#SBATCH --error=logs/learned_analytical_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=0-23  # Adjust based on number of edge types

# Learned Analytical Formula - HPC Batch Processing
# This script runs the learned analytical formula notebook for all edge types
# using papermill for parameterization and SLURM for distributed execution


# Exit if any command fails
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Get the directory of this script and define base paths relative to it
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"
logs_dir="${BASE_DIR}/logs/learned_analytical"
results_dir = "${BASE_DIR}/results"

# Create logs directory if it doesn't exist
mkdir -p $logs_dir
mkdir -p $results_dir

echo "****** Running model comparison analysis for job array index $SLURM_ARRAY_TASK_ID ******"
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


# Check if edge type exists
EDGE_FILE="$data_dir/permutations/000.hetmat/edges/${EDGE_TYPE}.sparse.npz"
if [ ! -f "$EDGE_FILE" ]; then
    echo "ERROR: Edge file not found: $EDGE_FILE"
    echo "Skipping $EDGE_TYPE"
    exit 1
fi

# Check if empirical frequencies exist
EMPIRICAL_FILE="$results_dir/empirical_edge_frequencies/edge_frequency_by_degree_${EDGE_TYPE}.csv"
if [ ! -f "$EMPIRICAL_FILE" ]; then
    echo "ERROR: Empirical frequency file not found: $EMPIRICAL_FILE"
    echo "Run notebook 3 (edge_frequency_by_degree) first"
    exit 1
fi


# Papermill parameters
N_CANDIDATES="[2, 3, 5, 7, 10, 15, 20, 30, 40, 50]"
CONVERGENCE_THRESHOLD="0.02"
TARGET_METRIC="correlation"

# Set minimum metric based on edge type density
# Sparse graphs (<3% density): 0.95
# Dense graphs (>5% density): 0.98
# We'll use 0.95 as default and let the algorithm find the optimal N
MIN_METRIC_VALUE="0.95"

echo ""
echo "Parameters:"
echo "  Edge type: $EDGE_TYPE"
echo "  N candidates: $N_CANDIDATES"
echo "  Convergence threshold: $CONVERGENCE_THRESHOLD"
echo "  Target metric: $TARGET_METRIC >= $MIN_METRIC_VALUE"
echo ""

# Output notebook path
mkdir -p "$notebooks_dir/outputs/learned_analytical"
OUTPUT_NOTEBOOK="$notebooks_dir/outputs/learned_analytical/8_${EDGE_TYPE}_learned_analytical_executed.ipynb"

# Run notebook with papermill
echo "Executing notebook with papermill..."
echo "------------------------------------------------------------------------"

papermill \
    "$NOTEBOOKS_DIR/8_learned_analytical_formula.ipynb" \
    "$OUTPUT_NOTEBOOK" \
    -p edge_type "$EDGE_TYPE" \
    -p N_candidates "$N_CANDIDATES" \
    -p convergence_threshold "$CONVERGENCE_THRESHOLD" \
    -p target_metric "$TARGET_METRIC" \
    -p min_metric_value "$MIN_METRIC_VALUE" \
    --log-output \
    --progress-bar

PAPERMILL_EXIT_CODE=$?

echo ""
echo "------------------------------------------------------------------------"
if [ $PAPERMILL_EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Notebook executed successfully"
    echo "  Output: $OUTPUT_NOTEBOOK"

    # Check if results were created
    RESULTS_FILE="$results_dir/${EDGE_TYPE}_results/${EDGE_TYPE}_learned_predictions.csv.gz"
    if [ -f "$RESULTS_FILE" ]; then
        echo "  Predictions: $RESULTS_FILE"
        FILE_SIZE=$(du -h "$RESULTS_FILE" | cut -f1)
        echo "  File size: $FILE_SIZE"
    fi
else
    echo "✗ ERROR: Notebook execution failed with exit code $PAPERMILL_EXIT_CODE"
    echo "  Check logs: $LOGS_DIR/learned_analytical_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
fi

echo ""
echo "========================================================================"
echo "End time: $(date)"
echo "========================================================================"

exit $PAPERMILL_EXIT_CODE
