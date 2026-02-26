#!/bin/bash
#SBATCH --job-name=min_perm_summary
#SBATCH --output=../logs/07_minimum_permutations_summary_%j.out
#SBATCH --error=../logs/07_minimum_permutations_summary_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=amilan
#SBATCH --qos=normal

# Minimum Permutations Summary
# Aggregates results from all 24 edge types (notebook 6)
# Must run AFTER all notebook 6 jobs complete

set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Get the directory of this script and define base paths relative to it
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"
logs_dir="${BASE_DIR}/logs"
results_dir="${BASE_DIR}/results/minimum_permutations_ml_summary"

# Create logs and results directories if they don't exist
mkdir -p $logs_dir
mkdir -p $results_dir

echo "====================================================================="
echo "Minimum Permutations Summary"
echo "====================================================================="
echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Define paths
INPUT_NOTEBOOK="${notebooks_path}/07_minimum_permutations_summary.ipynb"
OUTPUT_NOTEBOOK="${notebooks_path}/executed/07_minimum_permutations_summary_executed.ipynb"

echo "Input notebook: $INPUT_NOTEBOOK"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Results directory: $results_dir"
echo ""

# Check if input notebook exists
if [ ! -f "$INPUT_NOTEBOOK" ]; then
    echo "ERROR: Input notebook not found: $INPUT_NOTEBOOK"
    exit 1
fi

# Check if any results from notebook 6 exist
RESULTS_COUNT=$(find "${BASE_DIR}/results/minimum_permutations_ml" -name "*_summary.json" 2>/dev/null | wc -l)
echo "Found $RESULTS_COUNT edge type results from notebook 6"

if [ "$RESULTS_COUNT" -lt 1 ]; then
    echo "WARNING: No results from notebook 6 found!"
    echo "Expected results in: ${BASE_DIR}/results/minimum_permutations_ml/"
    echo "Run notebook 6 (script 06) for all edge types first"
fi

echo ""
echo "Starting papermill execution..."
echo "---------------------------------------------------------------------"

# Run papermill
papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    --log-output \
    --progress-bar

PAPERMILL_EXIT_CODE=$?

echo ""
echo "---------------------------------------------------------------------"
if [ $PAPERMILL_EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Notebook executed successfully"
    echo "  Output: $OUTPUT_NOTEBOOK"

    # Display key results if they exist
    N_MIN_FILE="${results_dir}/N_min_by_edge_type.csv"
    if [ -f "$N_MIN_FILE" ]; then
        echo ""
        echo "N_min by edge type (first 10 rows):"
        head -n 11 "$N_MIN_FILE" | column -t -s,
    fi

    MODEL_STATS="${results_dir}/model_statistics.csv"
    if [ -f "$MODEL_STATS" ]; then
        echo ""
        echo "Model statistics:"
        cat "$MODEL_STATS" | column -t -s,
    fi
else
    echo "✗ ERROR: Notebook execution failed with exit code $PAPERMILL_EXIT_CODE"
    echo "  Check logs: ${logs_dir}/07_minimum_permutations_summary_${SLURM_JOB_ID}.err"
fi

echo ""
echo "====================================================================="
echo "End time: $(date)"
echo "====================================================================="

exit $PAPERMILL_EXIT_CODE
