#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err
#SBATCH --qos=normal

# Load modules
module load python/3.9
module load anaconda

# Activate conda environment
source activate hetionet_env

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create all directories
mkdir -p results/null_models
mkdir -p results/compositional_null
mkdir -p results/metapath_nulls
mkdir -p logs
mkdir -p notebooks/executed

echo "=========================================="
echo "FULL NULL MODEL PIPELINE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "Start time: $(date)"
echo ""

# Step 1: Train null models
echo "===== STEP 1: Training Null Models ====="
papermill notebooks/13_null_model_training.ipynb \
    notebooks/executed/13_null_model_training_executed.ipynb \
    --log-output \
    --progress-bar

if [ $? -eq 0 ]; then
    echo "✓ Null model training successful"
else
    echo "✗ Null model training failed"
    exit 1
fi
echo ""

# Step 2: Compositional null validation
echo "===== STEP 2: Compositional Null Validation ====="
papermill notebooks/14_fast_compositional_null.ipynb \
    notebooks/executed/14_fast_compositional_null_executed.ipynb \
    --log-output \
    --progress-bar

if [ $? -eq 0 ]; then
    echo "✓ Compositional null validation successful"
else
    echo "✗ Compositional null validation failed"
    exit 1
fi
echo ""

# Step 3: Generate metapath nulls
echo "===== STEP 3: Metapath Null Generation ====="
papermill notebooks/15_metapath_null_distributions.ipynb \
    notebooks/executed/15_metapath_null_distributions_executed.ipynb \
    --log-output \
    --progress-bar

if [ $? -eq 0 ]; then
    echo "✓ Metapath null generation successful"
else
    echo "✗ Metapath null generation failed"
    exit 1
fi
echo ""

# Step 4: Fix and run notebook 5
echo "===== STEP 4: Notebook 5 Fix ====="
papermill notebooks/5_model_testing_summary.ipynb \
    notebooks/executed/5_model_testing_summary_executed.ipynb \
    --log-output \
    --progress-bar

if [ $? -eq 0 ]; then
    echo "✓ Notebook 5 execution successful"
else
    echo "✗ Notebook 5 execution failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "PIPELINE COMPLETE"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Null models: results/null_models/"
echo "  - Compositional nulls: results/compositional_null/"
echo "  - Metapath nulls: results/metapath_nulls/"
echo "  - Summary plots: results/model_comparison_summary/"
echo ""
echo "Executed notebooks: notebooks/executed/"
