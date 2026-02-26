#!/bin/bash
#SBATCH --job-name=hetio_pair_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=amilan
#SBATCH --output=logs/pvalue_validation/hetio_pair_stats_%j.out
#SBATCH --error=logs/pvalue_validation/hetio_pair_stats_%j.err
#SBATCH --qos=normal

# Stage 1b: Compute DWPC and null stats for het.io pairs
#
# This script runs on HPC to compute DWPCs and null distribution statistics
# for pairs extracted from het.io data (using 29a_extract_hetio_pairs.py locally).
#
# Prerequisites:
#   1. Run locally: python scripts/29a_extract_hetio_pairs.py --n_pairs 100
#   2. Upload data/hetio_pairs_for_validation.csv to HPC
#   3. Submit this job: sbatch scripts/29b_compute_hetio_pair_stats.sh
#
# After completion:
#   1. Download results/pvalue_validation/hetio_pair_stats_20perms.csv
#   2. Run locally: python scripts/29c_compare_to_hetio.py --input_file <downloaded_file>
#
# Usage:
#   sbatch scripts/29b_compute_hetio_pair_stats.sh [N_PERMS] [N_SAMPLES]
#
# Arguments (optional):
#   N_PERMS:   Number of permutations for null distribution (default: 20)
#   N_SAMPLES: Samples per degree group per permutation (default: 50)

# Parse arguments
N_PERMS=${1:-20}
N_SAMPLES=${2:-50}

echo "===== Het.io Pair Stats Computation ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "N permutations: ${N_PERMS}"
echo "N samples per group: ${N_SAMPLES}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 16GB"
echo "Start time: $(date)"
echo ""

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CAPP_PROJECT_ROOT=$(pwd)

# Create output directories
mkdir -p results/pvalue_validation
mkdir -p logs/pvalue_validation

# Check input file exists
INPUT_FILE="data/hetio_pairs_for_validation.csv"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please run locally first: python scripts/29a_extract_hetio_pairs.py"
    echo "Then upload the file to HPC."
    exit 1
fi

echo "Input file found: $INPUT_FILE"
echo "Number of pairs: $(wc -l < $INPUT_FILE)"
echo ""

echo "Running het.io pair stats computation with ${N_PERMS} permutations..."
echo ""

# Run the computation script
python scripts/29b_compute_hetio_pair_stats.py \
    --n_perms ${N_PERMS} \
    --n_samples ${N_SAMPLES} \
    --input_file ${INPUT_FILE} \
    --output_file results/pvalue_validation/hetio_pair_stats_${N_PERMS}perms.csv

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Computation completed successfully"
    echo ""
    echo "Output file: results/pvalue_validation/hetio_pair_stats_${N_PERMS}perms.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Download: scp <hpc>:$(pwd)/results/pvalue_validation/hetio_pair_stats_${N_PERMS}perms.csv ."
    echo "  2. Run locally: python scripts/29c_compare_to_hetio.py --input_file hetio_pair_stats_${N_PERMS}perms.csv"
else
    echo "Computation failed with exit code ${EXIT_CODE}"
fi

echo "===== Complete ====="

exit $EXIT_CODE
