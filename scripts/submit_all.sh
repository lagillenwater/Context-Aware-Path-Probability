#!/bin/bash

echo "=========================================="
echo "Submitting HPC Jobs for Null Model Pipeline"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p results/null_models
mkdir -p results/compositional_null
mkdir -p results/metapath_nulls
mkdir -p notebooks/executed

# Option 1: Submit full pipeline (sequential)
echo "Submit option 1: Full pipeline (sequential, 8 hours)"
echo "  sbatch scripts/run_full_pipeline_hpc.sh"
echo ""

# Option 2: Submit individual jobs with dependencies
echo "Submit option 2: Individual jobs with dependencies"
echo ""

# Submit null training
JOB1=$(sbatch --parsable scripts/run_null_training_hpc.sh)
echo "  Submitted null training: Job ID $JOB1"

# Submit compositional null (depends on null training)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_compositional_null_hpc.sh)
echo "  Submitted compositional null: Job ID $JOB2 (depends on $JOB1)"

# Submit metapath array (depends on compositional null)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 scripts/run_metapath_array_hpc.sh)
echo "  Submitted metapath array: Job ID $JOB3 (depends on $JOB2)"

# Submit notebook 5 fix (depends on all above)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 scripts/run_notebook5_fix_hpc.sh)
echo "  Submitted notebook 5 fix: Job ID $JOB4 (depends on $JOB3)"

echo ""
echo "=========================================="
echo "Jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  bash scripts/monitor_jobs.sh"
echo ""
echo "Or watch queue:"
echo "  watch -n 10 'squeue -u \$USER'"
