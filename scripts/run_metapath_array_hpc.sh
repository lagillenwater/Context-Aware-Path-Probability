#!/bin/bash
#SBATCH --job-name=metapath_array
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --output=logs/metapath_array_%A_%a.out
#SBATCH --error=logs/metapath_array_%A_%a.err
#SBATCH --qos=normal

# Load modules
module load python/3.9
module load anaconda

# Activate conda environment
source activate hetionet_env

# Define metapaths (one per array task)
METAPATHS=(
    "CbGpPW"
    "CtDaG"
    "CbGaD"
    "CbGiGpPW"
    "CtDuG"
    "CbGdD"
    "CpDaG"
    "CrCbG"
    "CbGpBP"
    "CbGpCC"
)

# Get metapath for this array task
METAPATH=${METAPATHS[$SLURM_ARRAY_TASK_ID]}

echo "===== Processing Metapath: $METAPATH ====="
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Start time: $(date)"

# Run notebook with this specific metapath
papermill notebooks/15_metapath_null_distributions.ipynb \
    notebooks/executed/15_metapath_${METAPATH}_executed.ipynb \
    -p metapath_pattern "$METAPATH" \
    --log-output

echo "End time: $(date)"
echo "===== $METAPATH Complete ====="
