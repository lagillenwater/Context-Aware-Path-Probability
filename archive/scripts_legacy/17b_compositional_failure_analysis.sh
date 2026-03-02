#!/bin/bash
#SBATCH --job-name=17b_failure
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=../logs/17b_%j.out
#SBATCH --error=../logs/17b_%j.err

# Compositional Failure Analysis (BASELINE for comparison)
#
# Standardized Resources: 4 cores, 32GB, 6 hours
# Used as baseline to compare against degree signature models

# Load environment
module purge
module load anaconda
conda activate CAPP

# Set paths
REPO_DIR=/projects/$USER/repositories/Context-Aware-Path-Probability
cd $REPO_DIR

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "17b: Compositional Failure Analysis"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Cores: 4 (STANDARD)"
echo "  Memory: 32GB (STANDARD)"
echo "=========================================="

# Check notebook 17 results exist (optional check)
VALIDATION_FILE="results/compositional_validation/validation_summary.json"
if [ ! -f "$VALIDATION_FILE" ]; then
    echo "WARNING: Validation file not found: $VALIDATION_FILE"
    echo "Notebook 17 may not have been run yet, but continuing anyway..."
    echo "This notebook can run independently."
else
    echo "Notebook 17 results found: $VALIDATION_FILE"
fi

# Create results directory if needed
mkdir -p results/compositional_validation/plots

# DELETE ALL CSV FILES (including corrupted ones)
echo "Cleaning up old CSV files..."
rm -f results/compositional_validation/failure_analysis*.csv
echo "Deleted any existing failure_analysis CSV files"
echo ""

# Run notebook
papermill \
    notebooks/17b_compositional_failure_analysis.ipynb \
    notebooks/executed/17b_compositional_failure_analysis_executed.ipynb \
    --log-output

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: 17b completed"
    echo ""
    echo "Results saved to:"
    echo "  - results/compositional_validation/failure_analysis.csv"
    echo "  - results/compositional_validation/degree_stratified_correlations.csv"
    echo "  - results/compositional_validation/plots/*.png"
else
    echo "ERROR: 17b failed (exit code: $EXIT_CODE)"
fi

echo "Job completed at $(date)"
exit $EXIT_CODE
