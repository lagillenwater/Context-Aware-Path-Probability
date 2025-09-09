#!/usr/bin/sh

#SBATCH --job-name=3_generate_edge_pred_jobs
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_download_null_graphs.log
#SBATCH --error=../logs/error_download_null_graphs.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=8G

# Script to generate individual SLURM jobs for each edge type of each permuted network

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SCRIPT_DIR/..")
JOBS_DIR="${BASE_DIR}/slurm_jobs"
LOGS_DIR="${BASE_DIR}/logs"

# Create directories if they don't exist
mkdir -p "$JOBS_DIR"
mkdir -p "$LOGS_DIR"

echo "Generating SLURM jobs for edge prediction..."
echo "Base directory: $BASE_DIR"
echo "Jobs will be created in: $JOBS_DIR"
echo "Logs will be written to: $LOGS_DIR"

# Define edge type to node type mappings based on metagraph.json
declare -A ALL_EDGE_MAPPINGS=(
    # Anatomy edges
    ["AdG"]="Anatomy:Gene"
    ["AeG"]="Anatomy:Gene"
    ["AuG"]="Anatomy:Gene"
    
    # Compound edges
    ["CbG"]="Compound:Gene"
    ["CcSE"]="Compound:Side Effect"
    ["CdG"]="Compound:Gene"
    ["CpD"]="Compound:Disease"
    ["CrC"]="Compound:Compound"
    ["CtD"]="Compound:Disease"
    ["CuG"]="Compound:Gene"
    
    # Disease edges
    ["DaG"]="Disease:Gene"
    ["DdG"]="Disease:Gene"
    ["DlA"]="Disease:Anatomy"
    ["DpS"]="Disease:Symptom"
    ["DrD"]="Disease:Disease"
    ["DuG"]="Disease:Gene"
    
    # Gene edges
    ["GcG"]="Gene:Gene"
    ["GiG"]="Gene:Gene"
    ["GpBP"]="Gene:Biological Process"
    ["GpCC"]="Gene:Cellular Component"
    ["GpMF"]="Gene:Molecular Function"
    ["GpPW"]="Gene:Pathway"
    ["Gr>G"]="Gene:Gene"
    
    # Pharmacologic Class edges
    ["PCiC"]="Pharmacologic Class:Compound"
)

# For testing, limit to just 3 edge types
# To use all edge types, change the line below to: declare -A EDGE_MAPPINGS=("${ALL_EDGE_MAPPINGS[@]}")
declare -A EDGE_MAPPINGS=(
    ["AeG"]="Anatomy:Gene"
    ["CbG"]="Compound:Gene"
    ["DaG"]="Disease:Gene"
)

# Get all available permutations
PERMUTATIONS_DIR="${BASE_DIR}/data/permutations"
ALL_PERMUTATIONS=($(find "$PERMUTATIONS_DIR" -name "*.hetmat" -type d -exec basename {} \; | sort))

# For testing, limit to first 2 permutations
# To use all permutations, change the line below to: PERMUTATIONS=("${ALL_PERMUTATIONS[@]}")
PERMUTATIONS=("${ALL_PERMUTATIONS[@]:0:2}")

echo "Found ${#ALL_PERMUTATIONS[@]} total permutations: ${ALL_PERMUTATIONS[*]}"
echo "Testing with ${#PERMUTATIONS[@]} permutations: ${PERMUTATIONS[*]}"
echo "Found ${#ALL_EDGE_MAPPINGS[@]} total edge types, testing with ${#EDGE_MAPPINGS[@]}: ${!EDGE_MAPPINGS[*]}"

# Generate SLURM job scripts for each combination
TOTAL_JOBS=0
SUBMITTED_JOBS=0

for permutation in "${PERMUTATIONS[@]}"; do
    echo "Processing permutation: $permutation"
    
    for edge_type in "${!EDGE_MAPPINGS[@]}"; do
        # Parse source and target node types
        IFS=':' read -r source_type target_type <<< "${EDGE_MAPPINGS[$edge_type]}"
        
        # Create job name
        JOB_NAME="edge_pred_${permutation}_${edge_type}"
        
        # Define file paths
        JOB_SCRIPT="${JOBS_DIR}/${JOB_NAME}.sbatch"
        OUTPUT_LOG="${LOGS_DIR}/output_${JOB_NAME}.log"
        ERROR_LOG="${LOGS_DIR}/error_${JOB_NAME}.log"
        
        # Create SLURM job script
        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=amc-general
#SBATCH --output=${OUTPUT_LOG}
#SBATCH --error=${ERROR_LOG}
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Exit if any command fails
set -e

# Change to the base directory
cd "${BASE_DIR}"

# Create outputs directory if it doesn't exist
mkdir -p "notebooks/outputs"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

echo "=========================================="
echo "Starting edge prediction job"
echo "Permutation: ${permutation}"
echo "Edge Type: ${edge_type}"
echo "Source Node Type: ${source_type}"
echo "Target Node Type: ${target_type}"
echo "Job started at: \$(date)"
echo "Working directory: \$(pwd)"
echo "=========================================="

# Run the edge prediction notebook using papermill
papermill notebooks/3_learn_null_edge.ipynb \\
    notebooks/outputs/3_learn_null_edge_${permutation}_${edge_type}_output.ipynb \\
    -p permutations_subdirectory "permutations" \\
    -p permutation_name "${permutation}" \\
    -p output_dir "models" \\
    -p edge_type "${edge_type}" \\
    -p source_node_type "${source_type}" \\
    -p target_node_type "${target_type}"

echo "=========================================="
echo "Edge prediction notebook completed successfully"
echo "Job finished at: \$(date)"
echo "=========================================="
EOF

        chmod +x "$JOB_SCRIPT"
        ((TOTAL_JOBS++))
        
        echo "  Created job script: $JOB_SCRIPT"
        
        # Submit the job immediately
        JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
        if [ $? -eq 0 ]; then
            ((SUBMITTED_JOBS++))
            echo "  Submitted job: $JOB_ID"
        else
            echo "  ERROR: Failed to submit job: $JOB_SCRIPT"
        fi
    done
done

echo ""
echo "=========================================="
echo "TEST RUN - Job generation and submission completed!"
echo "Total jobs created: $TOTAL_JOBS"
echo "Total jobs submitted: $SUBMITTED_JOBS"
echo "Jobs directory: $JOBS_DIR"
echo "Note: This was a test run with only ${#PERMUTATIONS[@]} permutations and ${#EDGE_MAPPINGS[@]} edge types"
echo "To run all permutations and edge types, modify the arrays in the script"
echo "=========================================="
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To monitor job outputs:"
echo "  tail -f $LOGS_DIR/output_*.log"
echo ""
echo "To cancel all jobs:"
echo "  scancel -u \$USER"
