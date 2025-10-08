#!/bin/bash
#SBATCH --job-name=dp_dwpc_16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=amilan
#SBATCH --output=logs/dp_dwpc_16_%j.out
#SBATCH --error=logs/dp_dwpc_16_%j.err
#SBATCH --qos=normal

# =====================================
# Dynamic Programming DWPC HPC Script - Notebook 16
# =====================================
# Implements dynamic programming approach for DWPC calculation
# Methods: exact expectation, matrix DP, mean-field approximation
# Validates against empirical permutation values
# =====================================

echo "====================================="
echo "Starting Dynamic Programming DWPC Calculation (Notebook 16)"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "Time limit: 8 hours"
echo "Start time: $(date)"
echo "====================================="

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set Python unbuffered for real-time output
export PYTHONUNBUFFERED=1

# Create output directories
mkdir -p results/dynamic_programming
mkdir -p logs
mkdir -p notebooks/executed

# Check if required data files exist
echo "Checking data files..."
if [ -d "data/edges" ] && [ -d "data/permutations" ]; then
    echo "  ✅ Data directories found"
    echo "  Edge files:"
    ls -lh data/edges/*.tsv | head -3
    echo "  Permutation files:"
    ls -lh data/permutations/ | head -3
else
    echo "  ❌ ERROR: Required data directories not found!"
    echo "  Expected: data/edges/ and data/permutations/"
    exit 1
fi

# Check if hetmat files exist
echo "Checking hetmat files..."
if [ -d "data/hetmats" ]; then
    echo "  ✅ Hetmat directory found"
    ls -lh data/hetmats/*.tsv | head -3
else
    echo "  ⚠️ Warning: Hetmat directory not found - will be created if needed"
fi

# Run notebook with parameters
echo ""
echo "====================================="
echo "Running Dynamic Programming DWPC Notebook"
echo "====================================="

# Set parameters for HPC environment
PARAMS="--parameters test_metapath='CbGpPW' \
        --parameters damping_exponent=0.4 \
        --parameters max_permutations=1000 \
        --parameters batch_size=10000 \
        --parameters use_cache=True \
        --parameters validation_mode=True"

# Run with papermill
papermill notebooks/16_dynamic_programming_dwpc.ipynb \
    notebooks/executed/16_dynamic_programming_dwpc_executed.ipynb \
    $PARAMS \
    --log-output \
    --progress-bar \
    --kernel python3

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "✅ DYNAMIC PROGRAMMING DWPC COMPLETE!"
    echo "====================================="
    echo "Results saved to: results/dynamic_programming/"

    # List output files
    echo ""
    echo "Output files:"
    ls -lh results/dynamic_programming/ 2>/dev/null

    echo ""
    echo "Validation summary:"
    if [ -f "results/dynamic_programming/validation_summary.json" ]; then
        cat results/dynamic_programming/validation_summary.json
    else
        echo "  No validation summary found"
    fi

    echo ""
    echo "Method comparison:"
    if [ -f "results/dynamic_programming/method_comparison.csv" ]; then
        head -5 results/dynamic_programming/method_comparison.csv
    else
        echo "  No method comparison found"
    fi
else
    echo ""
    echo "====================================="
    echo "❌ ERROR: Notebook execution failed!"
    echo "====================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check error log for details"
    echo ""
    echo "Common issues:"
    echo "  1. Missing data files in data/edges/ or data/permutations/"
    echo "  2. Insufficient memory for large graphs"
    echo "  3. Network connectivity issues"
    echo "  4. Python environment problems"
fi

echo ""
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "====================================="

exit $EXIT_CODE