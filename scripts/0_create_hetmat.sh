#!/bin/sh

#SBATCH --job-name=create_hetmat
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_create_hetmat.log
#SBATCH --error=../logs/error_create_hetmat.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1 

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

echo "****** downloading and creating hetmat agecencies******"


input_notebook=${notebooks_path}/0_create-hetmat.ipynb
output_notebook=${notebooks_path}/0_create-hetmat.ipynb

papermill "$input_notebook" "$output_notebook" 


# Deactivate the virtual environment w/ conda
conda deactivate