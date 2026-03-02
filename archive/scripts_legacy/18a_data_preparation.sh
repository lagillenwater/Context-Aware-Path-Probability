#!/bin/bash
#SBATCH --job-name=18a_data_prep
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --array=1-8
#SBATCH --output=../logs/pathway_nn/data_prep/18a_data_prep_%A_%a.out
#SBATCH --error=../logs/pathway_nn/data_prep/18a_data_prep_%A_%a.err

# Data Preparation: Generate degree-binned training data from original graph
#
# Resources: 4 cores, 64GB RAM, 6 hours
# Memory: Higher due to pathway matrix computation
# Array: 8 metapaths processed in parallel

# Load environment
module purge
module load anaconda
conda activate CAPP

REPO_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")
cd $REPO_DIR

# Create directories
mkdir -p logs/pathway_nn/data_prep
mkdir -p results/pathway_nn/training_data

echo "=========================================="
echo "18a: Data Preparation"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Cores: 4"
echo "  Memory: 64GB"
echo "=========================================="

# Metapath definitions
declare -a METAPATHS=("CbGpPW" "CtDaG" "CbGaD" "CrCbG" "CbGiG" "CpDaG" "CbGpBP" "CbGpCC")
declare -a EDGE1=("CbG" "CtD" "CbG" "CrC" "CbG" "CpD" "CbG" "CbG")
declare -a EDGE2=("GpPW" "DaG" "GaD" "CbG" "GiG" "DaG" "GpBP" "GpCC")

# DEBUG MODE: Override metapath selection if DEBUG_METAPATH is set
if [ -n "$DEBUG_METAPATH" ]; then
    echo "DEBUG MODE: Using metapath $DEBUG_METAPATH"
    METAPATH="$DEBUG_METAPATH"
    # Find index of metapath
    for i in "${!METAPATHS[@]}"; do
        if [[ "${METAPATHS[$i]}" = "$METAPATH" ]]; then
            IDX=$i
            break
        fi
    done
    EDGE1_TYPE=${EDGE1[$IDX]}
    EDGE2_TYPE=${EDGE2[$IDX]}
else
    # Get metapath for this array task
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    METAPATH=${METAPATHS[$IDX]}
    EDGE1_TYPE=${EDGE1[$IDX]}
    EDGE2_TYPE=${EDGE2[$IDX]}
fi

echo "Processing metapath: $METAPATH"
echo "  Edge1: $EDGE1_TYPE"
echo "  Edge2: $EDGE2_TYPE"
echo ""

# Run notebook
papermill \
    notebooks/18a_data_preparation.ipynb \
    notebooks/executed/18a_data_preparation_${METAPATH}_executed.ipynb \
    -p metapath "$METAPATH" \
    -p edge1_type "$EDGE1_TYPE" \
    -p edge2_type "$EDGE2_TYPE" \
    --log-output

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Data preparation completed for $METAPATH"
    echo "Output: results/pathway_nn/training_data/${METAPATH}_training_data.csv"
else
    echo ""
    echo "ERROR: Data preparation failed for $METAPATH (exit code: $EXIT_CODE)"
fi

echo "Job completed at $(date)"
exit $EXIT_CODE
