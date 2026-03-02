#!/bin/bash
#SBATCH --job-name=compositional_validation
#SBATCH --output=../logs/compositional_validation/17_compositional_validation_%j.out
#SBATCH --error=../logs/compositional_validation/17_compositional_validation_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=amilan
#SBATCH --qos=normal

# Compositional Null Validation
#
# Purpose: Validate whether compositional calculation (E[path] = P(e1) × P(e2))
#          accurately predicts pathway counts in held-out permutations.
#
# Method:
#   1. Compute edge probabilities from permutations 1-20 (training)
#   2. Predict pathway counts via matrix multiplication (DP)
#   3. Compare to actual pathway counts in permutations 21-30 (validation)
#
# Decision criteria:
#   - r > 0.95: PASS → proceed to notebook 18
#   - 0.85 < r < 0.95: BIAS → consider corrections
#   - r < 0.85: FAIL → must use direct empirical

set -e

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Get the directory of this script and define base paths relative to it
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"
logs_dir="${BASE_DIR}/logs/compositional_validation"

# Create logs directory if it doesn't exist
mkdir -p $logs_dir
# Create logs directory
mkdir -p logs

# Configuration
INPUT_NOTEBOOK="${notebooks_path}/17_compositional_validation.ipynb"
OUTPUT_NOTEBOOK="${notebooks_path}/executed/17_compositional_validation_executed.ipynb"
TRAIN_START=1
TRAIN_END=20
VALID_START=21
VALID_END=30

echo "=================================================="
echo "Compositional Null Validation"
echo "=================================================="
echo "Input notebook: $INPUT_NOTEBOOK"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Training permutations: $TRAIN_START-$TRAIN_END"
echo "Validation permutations: $VALID_START-$VALID_END"
echo "=================================================="

# Run notebook with papermill
papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    -p train_perms_start "$TRAIN_START" \
    -p train_perms_end "$TRAIN_END" \
    -p valid_perms_start "$VALID_START" \
    -p valid_perms_end "$VALID_END" \
    -p random_seed 42

echo ""
echo "=================================================="
echo "Notebook execution complete!"
echo "Output: $OUTPUT_NOTEBOOK"
echo "Results: results/compositional_validation/"
echo "=================================================="

# Check validation results
if [ -f "results/compositional_validation/validation_summary.json" ]; then
    echo ""
    echo "Validation Summary:"
    echo "-------------------"
    cat results/compositional_validation/validation_summary.json | grep -E '"overall_decision"|"overall_mean_pearson_r"|"recommendation"'
    echo ""
fi
