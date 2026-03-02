#!/bin/bash
#SBATCH --job-name=14.1_edge_corr
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=../logs/14.1_edge_correlation_%j.out
#SBATCH --error=../logs/14.1_edge_correlation_%j.err

# Edge Type Correlation Analysis with Degree Coloring
module purge
module load anaconda
conda activate CAPP

REPO_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")
cd $REPO_DIR

mkdir -p logs results/edge_correlation_analysis

echo "=========================================="
echo "14.1: Edge Correlation Analysis"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Start time: $(date)"
echo "=========================================="

echo "Repository: $REPO_DIR"
echo "Available memory: ${SLURM_MEM_PER_NODE}MB"
echo "Available CPUs: $SLURM_NTASKS"

# Run the notebook with papermill
echo ""
echo "Executing edge correlation analysis notebook..."

papermill \
    notebooks/14.1_edge_correlation_analysis.ipynb \
    notebooks/executed/14.1_edge_correlation_analysis_executed.ipynb \
    --log-output

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Edge correlation analysis complete"
    echo "Results saved to: results/edge_correlation_analysis/"

    # Display quick summary if results exist
    if [ -f "results/edge_correlation_analysis/edge_correlation_results.csv" ]; then
        echo ""
        echo "Quick Summary:"
        echo "  Edge types analyzed: $(tail -n +2 results/edge_correlation_analysis/edge_correlation_results.csv | wc -l)"
        echo "  Results file: results/edge_correlation_analysis/edge_correlation_results.csv"
        if [ -f "results/edge_correlation_analysis/edge_correlation_analysis.png" ]; then
            echo "  Plot saved: results/edge_correlation_analysis/edge_correlation_analysis.png"
        fi
    fi
else
    echo "ERROR: Analysis failed with exit code $EXIT_CODE"
    echo "Check the notebook execution logs above for details."
fi

echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE