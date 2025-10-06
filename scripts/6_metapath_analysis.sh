#!/bin/sh

#SBATCH --job-name=metapath_analysis
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_metapath_analysis_%a.log
#SBATCH --error=../logs/error_metapath_analysis_%a.log
#SBATCH --time=03:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --array=1-5

# Master script to run metapath analysis for different metapath patterns
# This creates a job array where each job processes a different metapath

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

echo "****** Running metapath analysis for job array index $SLURM_ARRAY_TASK_ID ******"
echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Define array of metapath patterns to analyze
# Each entry contains: "metapath_pattern:edge_type1,edge_type2,edge_type3"
METAPATH_CONFIGS=(
    "CbGpPWpG:CbG,GpPW,GpPW"
    "CtDaGdG:CtD,DaG,DdG"
    "CbGiGdG:CbG,GiG,GdG"
    "CuGcGdG:CuG,GcG,GdG"
    "CdGpPWpG:CdG,GpPW,GpPW"
)

# Get the metapath configuration for this array task
METAPATH_CONFIG=${METAPATH_CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}
METAPATH_PATTERN=$(echo $METAPATH_CONFIG | cut -d':' -f1)
EDGE_TYPES_STR=$(echo $METAPATH_CONFIG | cut -d':' -f2)

echo "Processing metapath: $METAPATH_PATTERN"
echo "Edge types: $EDGE_TYPES_STR"

# Convert edge types string to array format for papermill
EDGE_TYPES="[\"$(echo $EDGE_TYPES_STR | sed 's/,/\", \"/g')\"]"

# Define paths
input_notebook="${notebooks_path}/6_metapath_probability_analysis.ipynb"
output_notebook="${notebooks_path}/outputs/metapath_analysis_notebooks/6_metapath_analysis_output_${METAPATH_PATTERN}.ipynb"

echo "Input notebook: $input_notebook"
echo "Output notebook: $output_notebook"

# Create output directory if it doesn't exist
mkdir -p "${notebooks_path}/outputs/metapath_analysis_notebooks"
echo "Created output directory: ${notebooks_path}/outputs/metapath_analysis_notebooks"

# Run papermill with the metapath parameters
echo "Starting papermill execution for metapath: $METAPATH_PATTERN"
papermill "$input_notebook" "$output_notebook" \
    -p metapath_pattern "$METAPATH_PATTERN" \
    -p edge_types "$EDGE_TYPES" \
    -p anomaly_threshold 0.05

echo "Metapath analysis for $METAPATH_PATTERN completed successfully at $(date)"