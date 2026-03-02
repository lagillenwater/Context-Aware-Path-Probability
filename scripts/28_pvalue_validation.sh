#!/bin/bash
#SBATCH --job-name=pval_validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=amilan
#SBATCH --output=logs/pvalue_validation/pvalue_validation_%j.out
#SBATCH --error=logs/pvalue_validation/pvalue_validation_%j.err
#SBATCH --qos=normal

# P-Value Validation Script
# Tests calibration of gamma-hurdle p-values using degree-grouped null distributions
#
# Usage:
#   sbatch scripts/28_pvalue_validation.sh [N_PERMS] [N_PAIRS] [N_SAMPLES]
#
# Arguments (optional, with defaults):
#   N_PERMS:   Number of permutations for null distribution (default: 20)
#   N_PAIRS:   Number of pairs for calibration test (default: 100)
#   N_SAMPLES: Samples per degree group per permutation (default: 50)
#
# Examples:
#   sbatch scripts/28_pvalue_validation.sh           # Uses defaults (20, 100, 50)
#   sbatch scripts/28_pvalue_validation.sh 20 100 50 # Explicit values
#   sbatch scripts/28_pvalue_validation.sh 50 200 100 # More thorough analysis

# Parse arguments with defaults
N_PERMS=${1:-20}
N_PAIRS=${2:-100}
N_SAMPLES=${3:-50}

echo "===== P-Value Validation ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "N permutations: ${N_PERMS}"
echo "N calibration pairs: ${N_PAIRS}"
echo "N samples per group: ${N_SAMPLES}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 16GB"
echo "Start time: $(date)"
echo ""

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CAPP_PROJECT_ROOT=$(pwd)

# Create output directories
mkdir -p results/pvalue_validation
mkdir -p logs/pvalue_validation

echo "Running p-value validation with ${N_PERMS} permutations..."
echo ""

# Run the validation script
# Note: Using --skip_hetio since Multi-DWPC data may not be available on HPC
python scripts/validate_pvalue_end_to_end.py \
    --n_perms ${N_PERMS} \
    --n_pairs ${N_PAIRS} \
    --n_samples ${N_SAMPLES} \
    --output_dir results/pvalue_validation \
    --skip_hetio

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Validation completed successfully"
    echo "Results saved to: results/pvalue_validation/"
    echo ""
    echo "Output files:"
    ls -la results/pvalue_validation/
else
    echo "Validation failed with exit code ${EXIT_CODE}"
fi

echo "===== Complete ====="

exit $EXIT_CODE
