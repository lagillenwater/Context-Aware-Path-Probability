#!/bin/sh

# Script to run the 3_download_null_graphs notebook using papermill
#
# Usage:
#   ./3_download_null_graphs.sh
#
# SLURM usage:
#   sbatch 3_download_null_graphs.sh
#
# This script downloads the pre-computed Hetionet permutations from GitHub
# and organizes them into the data/permutations/hetio200/ directory

#SBATCH --job-name=download_null_graphs
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_download_null_graphs.log
#SBATCH --error=../logs/error_download_null_graphs.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=8G

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Define relative data and output paths
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"

# Conda Environment:
module load anaconda
conda deactivate
conda activate dwpc_rnn

##########################################################################################################
##########################################################################################################

echo "****** Downloading Hetionet null graphs ******"

input_notebook=${notebooks_path}/3_download_null_graphs.ipynb
output_notebook=${notebooks_path}/outputs/3_download_null_graphs_output.ipynb

# Create outputs directory if it doesn't exist
mkdir -p "${notebooks_path}/outputs"

echo "Input notebook: $input_notebook"
echo "Output notebook: $output_notebook"

# Run the notebook using papermill
echo "Starting download of null graphs at $(date)"
papermill "$input_notebook" "$output_notebook" \
    --kernel dwpc_rnn \
    --log-output

echo "Hetionet null graphs download completed successfully!"

# Deactivate the virtual environment w/ conda
conda deactivate
