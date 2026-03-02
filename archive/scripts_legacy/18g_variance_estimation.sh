#!/bin/bash
#SBATCH --job-name=18g_variance_estimation
#SBATCH --output=logs/18g_variance_estimation_%A_%a.out
#SBATCH --error=logs/18g_variance_estimation_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10

# Notebook 18g: Variance Estimation from Permutations
#
# Purpose:
#   Validate trained degree signature neural network on 20 degree-preserving
#   permutations and compute variance estimates for anomaly detection.
#
# Dependencies:
#   - Notebook 18f must be completed (trained model)
#   - Notebook 18c must be completed (permutations 001-020 with pathways)
#
# Outputs:
#   - results/pathway_nn/variance_analysis/{metapath}_variance_estimates.csv
#   - results/pathway_nn/variance_analysis/{metapath}_permutation_metrics.csv
#   - results/pathway_nn/variance_analysis/{metapath}_validation_summary.json
#   - results/pathway_nn/variance_analysis/permutation_{id}_predictions.npy
#   - results/pathway_nn/variance_analysis/all_permutations_results.npz
#   - results/pathway_nn/variance_analysis/{metapath}_permutation_validation.png
#
# Usage:
#   sbatch scripts/18g_variance_estimation.sh

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
echo "Variance Estimation for ${METAPATH}"
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

# Create logs directory
mkdir -p logs

# Run notebook with papermill
papermill notebooks/18g_variance_estimation.ipynb \
    notebooks/executed/18g_variance_estimation_${METAPATH}_executed.ipynb \
    -p metapath "${METAPATH}" \
    -p edge1_type "${EDGE1_TYPE}" \
    -p edge2_type "${EDGE2_TYPE}" \
    -p n_permutations 20 \
    -p first_perm_id 1 \
    -p n_degree_bins 10 \
    -p n_inter_bins 10 \
    -p random_seed 42

echo ""
echo "========================================="
echo "Variance Estimation Completed"
echo "End time: $(date)"
echo "========================================="