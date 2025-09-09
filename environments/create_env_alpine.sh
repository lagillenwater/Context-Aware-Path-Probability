#!/bin/sh

#SBATCH --job-name=create_env
#SBATCH --account=amc-general
#SBATCH --output=../logs/create_env.log
#SBATCH --error=../logs/error_create_env.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1 

# Exit if any command fails
set -e

# Create the dwpc_rnn conda environment from environment.yml

## for alpine
module load anaconda

# create environment from file
conda env create -f environment.yml

echo "created environment: CAPP"
