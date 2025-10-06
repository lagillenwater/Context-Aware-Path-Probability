#!/bin/bash

#SBATCH --job-name=model_summary
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_model_summary.log
#SBATCH --error=../logs/error_model_summary.log
#SBATCH --time=02:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=32G

# Model Testing Summary
# This script runs notebook 5 to summarize model testing results from notebook 4 across all edge types

# Exit if any command fails
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Get the directory of this script and define base paths relative to it
notebooks_path="${BASE_DIR}/notebooks"
logs_dir="${BASE_DIR}/logs"
results_dir="${BASE_DIR}/results/model_comparison_summary"

# Create logs and results directories if they don't exist
mkdir -p $logs_dir
mkdir -p $results_dir

echo "****** Running model testing summary ******"
echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Define paths
input_notebook="${notebooks_path}/5_model_testing_summary.ipynb"
output_notebook="${notebooks_path}/outputs/5_model_testing_summary_executed.ipynb"

echo "Input notebook: $input_notebook"
echo "Output notebook: $output_notebook"

# Create output directory if it doesn't exist
mkdir -p "${notebooks_path}/outputs"
echo "Created output directory: ${notebooks_path}/outputs"

echo ""
echo "Executing notebook 5: Model Testing Summary"
echo "------------------------------------------------------------------------"

# Run papermill (no parameters needed - uses all edge types by default)
papermill "$input_notebook" "$output_notebook" \
    --log-output \
    --progress-bar

PAPERMILL_EXIT_CODE=$?

echo ""
echo "------------------------------------------------------------------------"
if [ $PAPERMILL_EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Notebook executed successfully"
    echo "  Output: $output_notebook"

    # List generated files
    if [ -d "$results_dir" ]; then
        echo "  Results directory: $results_dir"
        echo "  Generated files:"
        ls -lh "$results_dir" | tail -n +2 | awk '{printf "    %s (%s)\n", $9, $5}'
    fi
else
    echo "✗ ERROR: Notebook execution failed with exit code $PAPERMILL_EXIT_CODE"
    echo "  Check logs: $logs_dir/error_model_summary.log"
fi

echo ""
echo "========================================================================"
echo "End time: $(date)"
echo "========================================================================"

exit $PAPERMILL_EXIT_CODE
