#!/bin/bash
#SBATCH --job-name=18h_anomaly_detection
#SBATCH --output=logs/18h_anomaly_detection_%A_%a.out
#SBATCH --error=logs/18h_anomaly_detection_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

# Notebook 18h: Anomaly Detection with DWPC Comparison
#
# Purpose:
#   Detect anomalous compound-pathway pairs using degree-conditioned null model
#   and variance estimates. Compare anomaly scores to DWPC to identify novel
#   biological associations.
#
# Dependencies:
#   - Notebook 18f must be completed (trained model)
#   - Notebook 18g must be completed (variance estimates)
#   - Original Hetionet edges in data/edges/ or data/permutations/000.hetmat/edges/
#
# Outputs:
#   - results/pathway_nn/anomaly_detection/{metapath}_all_anomalies.csv
#   - results/pathway_nn/anomaly_detection/{metapath}_significant_anomalies.csv
#   - results/pathway_nn/anomaly_detection/{metapath}_anomalies_p005.csv
#   - results/pathway_nn/anomaly_detection/{metapath}_anomalies_p001.csv
#   - results/pathway_nn/anomaly_detection/{metapath}_anomalies_p0001.csv
#   - results/pathway_nn/anomaly_detection/{metapath}_novel_discoveries.csv
#   - results/pathway_nn/anomaly_detection/{metapath}_anomaly_summary.json
#   - results/pathway_nn/anomaly_detection/{metapath}_volcano_plot.png
#   - results/pathway_nn/anomaly_detection/{metapath}_dwpc_comparison.png
#   - results/pathway_nn/anomaly_detection/{metapath}_anomaly_distributions.png
#
# Usage:
#   sbatch scripts/18h_anomaly_detection.sh

set -e

# Activate conda environment
source /opt/homebrew/bin/conda
conda activate CAPP

# Define metapaths
METAPATHS=(
    "CbGpPW"
    "CbGaD"
    "CbGdD"
    "CbGiGpPW"
    "CbGpBP"
    "CbGpCC"
    "CpDaG"
    "CrCbG"
    "CtDaG"
    "CtDuG"
)

# Get metapath for this array task
METAPATH=${METAPATHS[$((SLURM_ARRAY_TASK_ID-1))]}

echo "========================================="
echo "Anomaly Detection for ${METAPATH}"
echo "========================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

# Extract edge types from metapath
# CbGpPW -> CbG and GpPW
EDGE1_TYPE=$(echo ${METAPATH} | sed -E 's/^([A-Z][a-z][A-Z]).*$/\1/')
EDGE2_TYPE=$(echo ${METAPATH} | sed -E 's/^[A-Z][a-z][A-Z](.*)$/\1/')

echo "Edge1 type: ${EDGE1_TYPE}"
echo "Edge2 type: ${EDGE2_TYPE}"
echo ""

# Check if trained model exists
MODEL_FILE="results/pathway_nn/trained_models/${METAPATH}_Degree_Sig_NN.pt"
if [ ! -f "${MODEL_FILE}" ]; then
    echo "ERROR: Trained model not found: ${MODEL_FILE}"
    echo "Please run notebook 18f first!"
    exit 1
fi

# Check if variance estimates exist
VARIANCE_FILE="results/pathway_nn/variance_analysis/${METAPATH}_variance_estimates.csv"
if [ ! -f "${VARIANCE_FILE}" ]; then
    echo "ERROR: Variance estimates not found: ${VARIANCE_FILE}"
    echo "Please run notebook 18g first!"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Run notebook with papermill
papermill notebooks/18h_anomaly_detection.ipynb \
    notebooks/executed/18h_anomaly_detection_${METAPATH}_executed.ipynb \
    -p metapath "${METAPATH}" \
    -p edge1_type "${EDGE1_TYPE}" \
    -p edge2_type "${EDGE2_TYPE}" \
    -p significance_threshold 0.01 \
    -p min_pathway_count 1 \
    -p n_degree_bins 10 \
    -p n_inter_bins 10 \
    -p dwpc_damping 0.4 \
    -p random_seed 42

echo ""
echo "========================================="
echo "Anomaly Detection Completed"
echo "End time: $(date)"
echo "========================================="