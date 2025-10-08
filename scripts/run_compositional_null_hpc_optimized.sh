#!/bin/bash
#SBATCH --job-name=comp_null_opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=amilan
#SBATCH --output=logs/comp_null_opt_%j.out
#SBATCH --error=logs/comp_null_opt_%j.err
#SBATCH --qos=normal

# =====================================
# Optimized Compositional Null HPC Script
# =====================================
# Key improvements:
# 1. Increased time limit: 8 hours (was 2 hours)
# 2. More CPUs: 16 (was 8) for parallel processing
# 3. More memory: 64GB (was 32GB) for large matrices
# 4. Uses optimized notebook with vectorization
# 5. Better error handling and checkpointing
# =====================================

echo "====================================="
echo "Starting Optimized Compositional Null Calculation"
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
mkdir -p results/compositional_null
mkdir -p logs
mkdir -p notebooks/executed

# Check if null models exist
echo "Checking null model files..."
if [ -d "results/null_models" ]; then
    echo "  ✅ Null models directory found"
    ls -lh results/null_models/*.pkl | head -5
else
    echo "  ❌ ERROR: Null models directory not found!"
    echo "  Please ensure null models are in results/null_models/"
    exit 1
fi

# Check if required edge files exist
echo "Checking data files..."
if [ -d "data/edges" ] || [ -d "data/permutations" ]; then
    echo "  ✅ Data directories found"
else
    echo "  ⚠️ Warning: Data directories may be missing"
fi

# Run optimized notebook with parameters
echo ""
echo "====================================="
echo "Running Optimized Notebook"
echo "====================================="

# Set parameters for HPC environment
PARAMS="--parameters test_metapath='CbGpPW' \
        --parameters model_type='rf' \
        --parameters chunk_size=50000 \
        --parameters use_cache=True \
        --parameters save_checkpoints=True"

# Run with papermill
papermill notebooks/14_fast_compositional_null_optimized.ipynb \
    notebooks/executed/14_fast_compositional_null_optimized_executed.ipynb \
    $PARAMS \
    --log-output \
    --progress-bar \
    --kernel python3

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "✅ COMPOSITIONAL NULL COMPLETE!"
    echo "====================================="
    echo "Results saved to: results/compositional_null/"

    # List output files
    echo ""
    echo "Output files:"
    ls -lh results/compositional_null/ 2>/dev/null
else
    echo ""
    echo "====================================="
    echo "❌ ERROR: Notebook execution failed!"
    echo "====================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check error log for details"
fi

echo ""
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "====================================="

exit $EXIT_CODE