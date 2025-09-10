#!/bin/sh

#SBATCH --job-name=learn_nulls
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_learn_permutation_nulls.log
#SBATCH --error=../logs/error_learn_permutation_nulls.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1 

# Master script to generate individual SLURM job scripts for learning null edge predictions
# This allows for distributed processing across multiple HPC nodes for each edge type and permutation

# Exit if any command fails
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Get the directory of this script and define base paths relative to it
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"
logs_dir="${BASE_DIR}/logs"
jobs_dir="${BASE_DIR}/scripts/null_learning_jobs"

# make jobs directory
mkdir -p $jobs_dir

echo "****** Generating SLURM job scripts for learning null edge predictions ******"

# Define edge types and their corresponding node types
declare -A edge_types=(
    ["AdG"]="Anatomy,Gene"
    ["AeG"]="Anatomy,Gene"
    ["AuG"]="Anatomy,Gene"
    ["CbG"]="Compound,Gene"
    ["CcSE"]="Compound,Side Effect"
    ["CdG"]="Compound,Gene"
    ["CpD"]="Compound,Disease"
    ["CrC"]="Compound,Compound"
    ["CtD"]="Compound,Disease"
    ["CuG"]="Compound,Gene"
    ["DaG"]="Disease,Gene"
    ["DdG"]="Disease,Gene"
    ["DlA"]="Disease,Anatomy"
    ["DpS"]="Disease,Symptom"
    ["DrD"]="Disease,Disease"
    ["DuG"]="Disease,Gene"
    ["GcG"]="Gene,Gene"
    ["GiG"]="Gene,Gene"
    ["GpBP"]="Gene,Biological Process"
    ["GpCC"]="Gene,Cellular Component"
    ["GpMF"]="Gene,Molecular Function"
    ["GpPW"]="Gene,Pathway"
    ["Gr>G"]="Gene,Gene"
    ["PCiC"]="Pharmacologic Class,Compound"
)

# Function to create individual SLURM job script for learning null edge prediction
create_null_learning_job() {
    local perm_num=$1
    local edge_type=$2
    local node_types=$3
    local source_node_type=$(echo $node_types | cut -d',' -f1)
    local target_node_type=$(echo $node_types | cut -d',' -f2)
    local job_script="${jobs_dir}/null_learning_${perm_num}_${edge_type}.sh"
    
    cat > "$job_script" << EOF
#!/bin/sh

#SBATCH --job-name=null_${perm_num}_${edge_type}
#SBATCH --account=amc-general
#SBATCH --output=../logs/output_null_learning_${perm_num}_${edge_type}.log
#SBATCH --error=../logs/error_null_learning_${perm_num}_${edge_type}.log
#SBATCH --time=02:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks-per-node=12
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

# Exit if any command fails
set -e

echo "Starting null edge learning for permutation ${perm_num}, edge type ${edge_type} at \$(date)"
echo "Running on node: \$SLURM_NODELIST"
echo "Job ID: \$SLURM_JOB_ID"
echo "Edge type: ${edge_type} (${source_node_type} -> ${target_node_type})"

# Load conda environment
module load anaconda
conda deactivate
conda activate CAPP

# Define paths
notebooks_path="${notebooks_path}"
input_notebook="\${notebooks_path}/3_learn_null_edge.ipynb"
output_notebook="\${notebooks_path}/outputs/null_learning_notebooks/3_learn_null_edge_output_${perm_num}_${edge_type}.ipynb"

echo "Input notebook: \$input_notebook"
echo "Output notebook: \$output_notebook"
echo "Permutation number: ${perm_num}"
echo "Edge type: ${edge_type}"
echo "Source node type: ${source_node_type}"
echo "Target node type: ${target_node_type}"

# Create output directory if it doesn't exist
mkdir -p "\${notebooks_path}/outputs/null_learning_notebooks"
echo "Created output directory: \${notebooks_path}/outputs/null_learning_notebooks"

# Run papermill with the required parameters
echo "Starting papermill execution..."
papermill "\$input_notebook" "\$output_notebook" \\
    -p permutation_name "${perm_num}" \\
    -p edge_type "${edge_type}" \\
    -p source_node_type "${source_node_type}" \\
    -p target_node_type "${target_node_type}" \\
    -p permutations_subdirectory "permutations" \\
    -p output_dir "models"

echo "Null edge learning for permutation ${perm_num}, edge type ${edge_type} completed successfully at \$(date)"
EOF

    # Make the job script executable
    chmod +x "$job_script"
    echo "Created job script: $job_script"
}

# Generate individual job scripts for permutations 0-10 and all edge types
echo "Generating job scripts for all permutations and edge types..."

for permutation_num in {0..10}; do
    echo "Processing permutation ${permutation_num}..."
    
    for edge_type in "${!edge_types[@]}"; do
        node_types="${edge_types[$edge_type]}"
        echo "  Creating job for edge type: ${edge_type} (${node_types})"
        create_null_learning_job $permutation_num $edge_type $node_types
    done
done

echo ""
echo "Job script generation completed!"
echo "Generated scripts for:"
echo "  - Permutations: 0-10 (11 total)"
echo "  - Edge types: ${#edge_types[@]} total"
echo "  - Total jobs: $((11 * ${#edge_types[@]}))"

# Automatically submit all jobs
echo ""
echo "Submitting all null learning jobs..."
JOB_IDS=()

for permutation_num in {0..10}; do
    echo "Submitting jobs for permutation ${permutation_num}..."
    
    for edge_type in "${!edge_types[@]}"; do
        job_script="${jobs_dir}/null_learning_${permutation_num}_${edge_type}.sh"
        echo "  Submitting ${edge_type}..."
        job_id=$(sbatch --parsable "$job_script")
        JOB_IDS+=($job_id)
        echo "    Job ID: $job_id"
    done
done

echo ""
echo "All jobs submitted! Total jobs: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor job status with: squeue -u \$USER"
echo "Cancel all jobs with: scancel ${JOB_IDS[@]}"
echo ""
echo "Check job progress in logs: ${logs_dir}/output_null_learning_*"
echo "Models will be saved to: ${BASE_DIR}/models/"
