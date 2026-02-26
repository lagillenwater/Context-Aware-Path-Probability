#!/bin/sh

#SBATCH --job-name=create_hetmat
#SBATCH --account=amc-general
#SBATCH --output=../logs/data_prep/hetmat/output_create_hetmat.log
#SBATCH --error=../logs/data_prep/hetmat/error_create_hetmat.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1 

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# On HPC runs, SLURM_SUBMIT_DIR points to the submission dir; locally fall back to script dir parent
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")
else
  BASE_DIR=$(realpath "$SCRIPT_DIR/..")
fi

# Define relative data and output paths
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"

# Conda environment (optional):
# If the HPC module system is available, load anaconda and activate the requested env.
# Locally, assume the caller has already activated the desired env (e.g., `conda activate CAPP`).
if command -v module >/dev/null 2>&1; then
  module load anaconda || true
  conda deactivate || true
  conda activate ${DWPC_ENV:-dwpc_rnn} || true
fi

##########################################################################################################
##########################################################################################################

echo "****** downloading and creating hetmat agecencies******"

input_notebook=${notebooks_path}/0_create-hetmat.ipynb
output_notebook=${notebooks_path}/0_create-hetmat.ipynb

papermill "$input_notebook" "$output_notebook" 

# Deactivate only if we activated in this script
if command -v module >/dev/null 2>&1; then
  conda deactivate || true
fi
