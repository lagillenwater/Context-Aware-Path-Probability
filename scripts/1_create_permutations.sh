#!/bin/sh

#SBATCH --job-name=permutation_array
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_permutation_%a.log
#SBATCH --error=../logs/error_permutation_%a.log
#SBATCH --time=00:30:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1
#SBATCH --mem=12G
#SBATCH --array=1-3

# Job array script for generating permutations
# Each array job processes one permutation using the SLURM_ARRAY_TASK_ID

# Exit if any command fails
set -e

# Get the permutation number from the array task ID
PERMUTATION_NUM=$SLURM_ARRAY_TASK_ID

echo "Starting permutation ${PERMUTATION_NUM} at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

notebooks_path="${BASE_DIR}/notebooks"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Define paths
input_notebook="${notebooks_path}/1_generate-permutations.ipynb"
output_notebook="${notebooks_path}/outputs/permutation_notebooks/1_generate-permutations_output_${PERMUTATION_NUM}.ipynb"

echo "Input notebook: $input_notebook"
echo "Output notebook: $output_notebook"
echo "Permutation number: ${PERMUTATION_NUM}"

# Create output directory if it doesn't exist
mkdir -p "${notebooks_path}/outputs/permutation_notebooks"
echo "Created output directory: ${notebooks_path}/outputs/permutation_notebooks"

# Run papermill with the permutation_number parameter
echo "Starting papermill execution..."
papermill "$input_notebook" "$output_notebook" -p permutation_number ${PERMUTATION_NUM}

echo "Permutation ${PERMUTATION_NUM} completed successfully at $(date)"

