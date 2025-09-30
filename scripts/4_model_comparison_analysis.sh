#!/bin/sh

#SBATCH --job-name=model_comparison
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_model_comparison_%a.log
#SBATCH --error=../logs/error_model_comparison_%a.log
#SBATCH --time=04:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1-24

# Master script to run model comparison analysis for all edge types
# This creates a job array where each job processes a different edge type

# Exit if any command fails
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Get the directory of this script and define base paths relative to it
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"
logs_dir="${BASE_DIR}/logs"

# Create logs directory if it doesn't exist
mkdir -p $logs_dir

echo "****** Running model comparison analysis for job array index $SLURM_ARRAY_TASK_ID ******"
echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Define array of edge files
EDGE_FILES=(
    "AdG.sparse.npz"
    "AeG.sparse.npz"
    "AuG.sparse.npz"
    "CbG.sparse.npz"
    "CcSE.sparse.npz"
    "CdG.sparse.npz"
    "CpD.sparse.npz"
    "CrC.sparse.npz"
    "CtD.sparse.npz"
    "CuG.sparse.npz"
    "DaG.sparse.npz"
    "DdG.sparse.npz"
    "DlA.sparse.npz"
    "DpS.sparse.npz"
    "DrD.sparse.npz"
    "DuG.sparse.npz"
    "GcG.sparse.npz"
    "GiG.sparse.npz"
    "GpBP.sparse.npz"
    "GpCC.sparse.npz"
    "GpMF.sparse.npz"
    "GpPW.sparse.npz"
    "Gr>G.sparse.npz"
    "PCiC.sparse.npz"
)

# Get the edge file for this array task
EDGE_FILE=${EDGE_FILES[$((SLURM_ARRAY_TASK_ID-1))]}
EDGE_TYPE=${EDGE_FILE%.sparse.npz}

echo "Processing edge type: $EDGE_TYPE"
echo "Edge file: $EDGE_FILE"

# Define paths
input_notebook="${notebooks_path}/4_model_testing_reorganized.ipynb"
output_notebook="${notebooks_path}/outputs/model_comparison_notebooks/4_model_testing_output_${EDGE_TYPE}.ipynb"

echo "Input notebook: $input_notebook"
echo "Output notebook: $output_notebook"

# Create output directory if it doesn't exist
mkdir -p "${notebooks_path}/outputs/model_comparison_notebooks"
echo "Created output directory: ${notebooks_path}/outputs/model_comparison_notebooks"

# Run papermill with the edge_file parameter
echo "Starting papermill execution for edge type: $EDGE_TYPE"
papermill "$input_notebook" "$output_notebook" -p edge_file "$EDGE_FILE" -p edge_type "$EDGE_TYPE"

echo "Model comparison analysis for $EDGE_TYPE completed successfully at $(date)"