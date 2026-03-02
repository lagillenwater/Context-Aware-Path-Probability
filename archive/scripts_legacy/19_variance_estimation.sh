#!/bin/bash
#SBATCH --job-name=variance_estimation
#SBATCH --output=logs/19_variance_estimation_%j.out
#SBATCH --error=logs/19_variance_estimation_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=amilan

# Variance Estimation for Pathway Nulls
#
# Purpose: Compute variance of pathway counts from held-out permutations
#          for z-score calculation in anomaly detection.
#
# Method:
#   1. Use permutations 21-30 (independent from training 1-20)
#   2. Compute pathway counts for each permutation
#   3. Calculate variance across permutations
#   4. Create adaptive degree bins for sparse pairs
#   5. Save variance matrices and lookup tables
#
# Dependencies: Notebook 17 must PASS

set -e

# Activate conda environment
source ~/.bashrc
conda activate CAPP

# Set up paths
REPO_DIR="$HOME/repositories/Context-Aware-Path-Probability"
cd "$REPO_DIR"

# Create logs directory
mkdir -p logs

# Configuration
INPUT_NOTEBOOK="notebooks/19_variance_estimation.ipynb"
OUTPUT_NOTEBOOK="notebooks/executed/19_variance_estimation_executed.ipynb"
VAR_START=21
VAR_END=30
MIN_SAMPLES=30

echo "=================================================="
echo "Variance Estimation for Pathway Nulls"
echo "=================================================="
echo "Input notebook: $INPUT_NOTEBOOK"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Variance permutations: $VAR_START-$VAR_END"
echo "Minimum samples per bin: $MIN_SAMPLES"
echo "=================================================="

# Check that notebook 17 validation passed
VALIDATION_FILE="results/compositional_validation/validation_summary.json"
if [ ! -f "$VALIDATION_FILE" ]; then
    echo "ERROR: Validation file not found: $VALIDATION_FILE"
    echo "Please run notebook 17_compositional_validation.ipynb first!"
    exit 1
fi

DECISION=$(grep -o '"overall_decision": "[^"]*"' "$VALIDATION_FILE" | cut -d'"' -f4)
if [ "$DECISION" == "FAILED" ]; then
    echo "ERROR: Compositional validation FAILED!"
    echo "Variance estimation not meaningful without compositional approach."
    exit 1
fi

echo "✓ Compositional validation check passed ($DECISION)"
echo ""

# Run notebook with papermill
papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    -p variance_perms_start "$VAR_START" \
    -p variance_perms_end "$VAR_END" \
    -p min_samples_per_bin "$MIN_SAMPLES" \
    -p random_seed 42

echo ""
echo "=================================================="
echo "Notebook execution complete!"
echo "Output: $OUTPUT_NOTEBOOK"
echo "Results: results/variance_estimates/"
echo "=================================================="

# Display summary
if [ -f "results/variance_estimates/summary.csv" ]; then
    echo ""
    echo "Variance Estimation Summary:"
    echo "----------------------------"
    head -n 20 results/variance_estimates/summary.csv
    echo ""
fi
