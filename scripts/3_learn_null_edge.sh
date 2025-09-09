#!/bin/sh

# Script to run the 2_learn_null_edge notebook with configurable permutations directory
#
# Usage:
#   ./2_learn_null_edge.sh [permutations_directory]
#
# Examples:
#   ./2_learn_null_edge.sh                                    # Uses default 'permutations' (local generated)
#   ./2_learn_null_edge.sh data/downloads/hetionet-permutations # Uses extracted download permutations
#   ./2_learn_null_edge.sh /path/to/custom/permutations       # Uses custom directory path
#
# SLURM usage:
#   sbatch 2_learn_null_edge.sh
#   sbatch 2_learn_null_edge.sh data/permutations/hetio200

#SBATCH --job-name=learn_null_edge
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_learn_null_edge_%A.log
#SBATCH --error=../logs/error_learn_null_edge_%A.log
#SBATCH --time=03:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1
#SBATCH --mem=32G

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Parse command line arguments
PERMUTATIONS_DIR="${1:-permutations}"  # Default to 'permutations' if no argument provided

# Define relative data and output paths
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"

# Conda Environment:
module load anaconda
conda deactivate
conda activate dwpc_rnn

##########################################################################################################
##########################################################################################################

echo "****** Running neural network edge prediction analysis ******"
echo "Using permutations directory: $PERMUTATIONS_DIR"

# Create a safe filename from the directory path for output naming
SAFE_DIR_NAME=$(echo "$PERMUTATIONS_DIR" | sed 's/[\/]/_/g' | sed 's/^data_//')

input_notebook=${notebooks_path}/2_learn_null_edge.ipynb
output_notebook=${notebooks_path}/outputs/2_learn_null_edge_output_${SAFE_DIR_NAME}.ipynb

# Create output directory if it doesn't exist
mkdir -p "${notebooks_path}/outputs"

echo "Input notebook: $input_notebook"
echo "Output notebook: $output_notebook"

# Run the notebook with papermill, passing the permutations parameter
papermill "$input_notebook" "$output_notebook" \
    --kernel dwpc_rnn \
    -p permutations_subdirectory "$PERMUTATIONS_DIR"

echo "Neural network edge prediction analysis completed successfully!"

# Deactivate the virtual environment w/ conda
conda deactivate
