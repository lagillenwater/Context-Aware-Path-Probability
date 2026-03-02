#!/bin/bash
#SBATCH --job-name=18d_negbin
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --array=1-8
#SBATCH --output=../logs/pathway_nn/training/18#SBATCH --output=../logs/18d_negbin_%A_%a.out
#SBATCH --error=../logs/pathway_nn/training/18#SBATCH --error=../logs/18d_negbin_%A_%a.err

# Train Negative Binomial GLM Model
module purge
module load anaconda
conda activate CAPP

REPO_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")
cd $REPO_DIR

mkdir -p logs/pathway_nn/training results/pathway_nn/trained_models results/pathway_nn/benchmarks

echo "=========================================="
echo "18d: Train NegBin GLM"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="

declare -a METAPATHS=("CbGpPW" "CtDaG" "CbGaD" "CrCbG" "CbGiG" "CpDaG" "CbGpBP" "CbGpCC")

# DEBUG MODE: Override metapath selection if DEBUG_METAPATH is set
if [ -n "$DEBUG_METAPATH" ]; then
    echo "DEBUG MODE: Using metapath $DEBUG_METAPATH"
    METAPATH="$DEBUG_METAPATH"
else
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    METAPATH=${METAPATHS[$IDX]}
fi

echo "Training NegBin GLM for $METAPATH"

papermill \
    notebooks/18d_train_negbin_glm.ipynb \
    notebooks/executed/18d_train_negbin_glm_${METAPATH}_executed.ipynb \
    -p metapath "$METAPATH" \
    --log-output

EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS: $METAPATH" || echo "ERROR: $METAPATH"
exit $EXIT_CODE
