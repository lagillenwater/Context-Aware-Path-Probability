#!/bin/bash
#SBATCH --job-name=anomaly_detection
#SBATCH --output=logs/20_anomaly_detection_%j.out
#SBATCH --error=logs/20_anomaly_detection_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=amilan

# Anomaly Detection in Hetionet
#
# Purpose: Identify metapaths with biological signal beyond degree structure
#          using z-scores.
#
# Method:
#   1. Load deployment recommendation from notebook 18 (best method)
#   2. Compute expected pathway counts using best method
#   3. Load variance estimates from notebook 19
#   4. Compute observed pathway counts in Hetionet
#   5. Calculate z-scores: z = (obs - exp) / sqrt(var)
#   6. Apply FDR correction
#   7. Identify significant anomalies
#
# Dependencies: Notebooks 17, 18, 19 must complete

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
INPUT_NOTEBOOK="notebooks/20_anomaly_detection.ipynb"
OUTPUT_NOTEBOOK="notebooks/executed/20_anomaly_detection_executed.ipynb"
FDR_THRESHOLD=0.05
MIN_COUNT=1

echo "=================================================="
echo "Anomaly Detection in Hetionet"
echo "=================================================="
echo "Input notebook: $INPUT_NOTEBOOK"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "FDR threshold: $FDR_THRESHOLD"
echo "Minimum pathway count: $MIN_COUNT"
echo "=================================================="

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# Check compositional validation
VALIDATION_FILE="results/compositional_validation/validation_summary.json"
if [ ! -f "$VALIDATION_FILE" ]; then
    echo "ERROR: Validation file not found: $VALIDATION_FILE"
    echo "Please run notebook 17_compositional_validation.ipynb first!"
    exit 1
fi

DECISION=$(grep -o '"overall_decision": "[^"]*"' "$VALIDATION_FILE" | cut -d'"' -f4)
if [ "$DECISION" == "FAILED" ]; then
    echo "ERROR: Compositional validation FAILED!"
    exit 1
fi
echo "  ✓ Compositional validation: $DECISION"

# Check deployment recommendation
RECOMMENDATION_FILE="results/null_approximation/deployment_recommendation.json"
if [ ! -f "$RECOMMENDATION_FILE" ]; then
    echo "ERROR: Deployment recommendation not found: $RECOMMENDATION_FILE"
    echo "Please run notebook 18_null_approximation_comparison.ipynb first!"
    exit 1
fi
echo "  ✓ Deployment recommendation found"

# Check variance estimates
VARIANCE_DIR="results/variance_estimates"
if [ ! -d "$VARIANCE_DIR" ]; then
    echo "ERROR: Variance estimates directory not found: $VARIANCE_DIR"
    echo "Please run notebook 19_variance_estimation.ipynb first!"
    exit 1
fi

N_VARIANCE=$(ls "$VARIANCE_DIR"/*_variance.npz 2>/dev/null | wc -l)
echo "  ✓ Variance estimates found: $N_VARIANCE metapaths"

echo ""
echo "All prerequisites satisfied!"
echo ""

# Run notebook with papermill
papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    -p fdr_threshold "$FDR_THRESHOLD" \
    -p min_pathway_count "$MIN_COUNT" \
    -p random_seed 42

echo ""
echo "=================================================="
echo "Notebook execution complete!"
echo "Output: $OUTPUT_NOTEBOOK"
echo "Results: results/anomaly_detection/"
echo "=================================================="

# Display summary
if [ -f "results/anomaly_detection/summary_by_metapath.csv" ]; then
    echo ""
    echo "Summary by Metapath:"
    echo "-------------------"
    head -n 15 results/anomaly_detection/summary_by_metapath.csv | column -t -s,
    echo ""
fi

echo ""
echo "=================================================="
echo "FULL VALIDATION PIPELINE COMPLETE!"
echo "=================================================="
echo ""
echo "Pipeline notebooks:"
echo "  ✓ 17_compositional_validation"
echo "  ✓ 18_null_approximation_comparison"
echo "  ✓ 19_variance_estimation"
echo "  ✓ 20_anomaly_detection"
echo ""
echo "Key results:"
echo "  - results/compositional_validation/validation_summary.json"
echo "  - results/null_approximation/deployment_recommendation.json"
echo "  - results/variance_estimates/summary.csv"
echo "  - results/anomaly_detection/significant_FDR05.csv"
echo ""
echo "=================================================="
