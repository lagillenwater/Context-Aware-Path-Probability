#!/bin/bash
#SBATCH --job-name=degree_analysis
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=1-24
#SBATCH --output=../logs/degree_analysis_%A_%a.out
#SBATCH --error=../logs/degree_analysis_%A_%a.err

# Degree-Based Error Analysis HPC Deployment Script
#
# This script runs comprehensive degree-based error analysis across all edge types
# using job arrays for parallel execution on HPC systems.
#
# Usage:
#   sbatch scripts/degree_analysis_hpc.sh
#
# Requirements:
#   - Completed model testing results for all edge types
#   - Empirical frequency data
#   - Python environment with required packages

# Setup
module purge
module load python/3.8.5
module load gcc/10.3.0

# Activate environment (adjust path as needed)
source ~/miniconda3/envs/hetionet/bin/activate

# Navigate to repository
cd $SLURM_SUBMIT_DIR/..

# Create output directories
mkdir -p results/degree_analysis_hpc
mkdir -p logs

# Define edge types array
EDGE_TYPES=(
    "AdG" "AeG" "AuG" "CbG" "CcSE" "CdG" "CpD" "CrC" "CtD" "CuG"
    "DaG" "DdG" "DlA" "DpS" "DrD" "DuG" "GcG" "GiG" "GpBP" "GpCC"
    "GpMF" "GpPW" "Gr>G" "PCiC"
)

# Get edge type for this array job (1-indexed)
EDGE_TYPE=${EDGE_TYPES[$SLURM_ARRAY_TASK_ID-1]}

echo "=========================================="
echo "Starting degree analysis for: $EDGE_TYPE"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../src:$PYTHONPATH"

# Run degree analysis pipeline
python3 << EOF
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
repo_dir = Path.cwd()
src_dir = repo_dir / 'src'
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'model_comparison'
output_dir = repo_dir / 'results' / 'degree_analysis_hpc'

sys.path.append(str(src_dir))

# Import modules
from degree_analysis import run_degree_analysis_pipeline

# Configuration
edge_type = "${EDGE_TYPE}"
small_graph_mode = False  # Full-scale analysis for HPC

print(f"Running degree analysis for {edge_type}...")

try:
    # Run complete pipeline
    file_paths = run_degree_analysis_pipeline(
        edge_type=edge_type,
        data_dir=data_dir,
        results_dir=results_dir,
        output_dir=output_dir,
        small_graph_mode=small_graph_mode
    )

    if file_paths:
        print(f"Success! Generated files for {len(file_paths)} models")
        for model, paths in file_paths.items():
            print(f"  {model}: {len(paths)} files")
    else:
        print(f"Warning: No output generated for {edge_type}")
        sys.exit(1)

except Exception as e:
    print(f"Error processing {edge_type}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"Degree analysis completed successfully for {edge_type}")
EOF

# Check exit status
if [ $? -eq 0 ]; then
    echo "✓ Degree analysis completed successfully for $EDGE_TYPE"
else
    echo "✗ Degree analysis failed for $EDGE_TYPE"
    exit 1
fi

echo "Job completed at: $(date)"