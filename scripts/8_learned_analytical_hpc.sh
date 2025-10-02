#!/bin/bash
#SBATCH --job-name=learned_analytical
#SBATCH --output=logs/learned_analytical_%A_%a.out
#SBATCH --error=logs/learned_analytical_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-23  # Adjust based on number of edge types

# Learned Analytical Formula - HPC Batch Processing
# This script runs the learned analytical formula notebook for all edge types
# using papermill for parameterization and SLURM for distributed execution

echo "========================================================================"
echo "Learned Analytical Formula Analysis"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Setup paths
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOKS_DIR="$REPO_DIR/notebooks"
RESULTS_DIR="$REPO_DIR/results/learned_analytical"
LOGS_DIR="$REPO_DIR/logs"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# List of edge types (24 total - adjust as needed)
EDGE_TYPES=(
    "AeG" "AdG" "AuG" "CbG" "CcSE" "CdG" "CpD" "Cr>G" "CrC" "CtD"
    "CuG" "DaG" "DdG" "DpS" "DrD" "DuG" "GcG" "GiG" "GpBP" "GpCC"
    "GpMF" "GpPW" "Gr>G" "PCiC"
)

# Get edge type for this array task
EDGE_TYPE="${EDGE_TYPES[$SLURM_ARRAY_TASK_ID]}"

echo "Processing edge type: $EDGE_TYPE"
echo "------------------------------------------------------------------------"

# Check if edge type exists
EDGE_FILE="$REPO_DIR/data/permutations/000.hetmat/edges/${EDGE_TYPE}.sparse.npz"
if [ ! -f "$EDGE_FILE" ]; then
    echo "ERROR: Edge file not found: $EDGE_FILE"
    echo "Skipping $EDGE_TYPE"
    exit 1
fi

# Check if empirical frequencies exist
EMPIRICAL_FILE="$REPO_DIR/results/empirical_edge_frequencies/edge_frequency_by_degree_${EDGE_TYPE}.csv"
if [ ! -f "$EMPIRICAL_FILE" ]; then
    echo "ERROR: Empirical frequency file not found: $EMPIRICAL_FILE"
    echo "Run notebook 3 (edge_frequency_by_degree) first"
    exit 1
fi

# Load conda/module environment (adjust for your HPC)
# module load python/3.10
# source activate CAPP
# OR
# module load anaconda
# conda activate CAPP

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
OUTPUT_NOTEBOOK="$RESULTS_DIR/${EDGE_TYPE}_learned_analytical_executed.ipynb"

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
    RESULTS_FILE="$RESULTS_DIR/${EDGE_TYPE}_results/${EDGE_TYPE}_learned_predictions.csv.gz"
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
