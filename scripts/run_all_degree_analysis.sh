#!/bin/bash

# Master script to run all degree-based analysis components
#
# This script coordinates the execution of all enhanced analysis components
# with proper dependencies and job chaining for HPC systems.
#
# Usage:
#   bash scripts/run_all_degree_analysis.sh

echo "========================================================"
echo "DEGREE-BASED ERROR ANALYSIS - MASTER DEPLOYMENT SCRIPT"
echo "========================================================"

# Ensure directories exist
mkdir -p logs
mkdir -p results/degree_analysis_hpc
mkdir -p results/enhanced_model_testing
mkdir -p results/learned_analytical_enhanced_hpc

# Function to submit job and get job ID
submit_job() {
    local script=$1
    local job_name=$2
    echo "Submitting $job_name..."
    job_id=$(sbatch --parsable $script)
    echo "  Job ID: $job_id"
    echo $job_id
}

# Function to submit dependent job
submit_dependent_job() {
    local script=$1
    local job_name=$2
    local dependency=$3
    echo "Submitting $job_name (depends on $dependency)..."
    job_id=$(sbatch --parsable --dependency=afterok:$dependency $script)
    echo "  Job ID: $job_id"
    echo $job_id
}

echo ""
echo "Phase 1: Core Degree Analysis"
echo "=============================="
degree_job=$(submit_job "scripts/degree_analysis_hpc.sh" "Degree Analysis")

echo ""
echo "Phase 2: Enhanced Model Testing (parallel with Phase 1)"
echo "======================================================"
model_job=$(submit_job "scripts/enhanced_model_testing_hpc.sh" "Enhanced Model Testing")

echo ""
echo "Phase 3: Enhanced Learned Formula Analysis (depends on Phase 1)"
echo "=============================================================="
formula_job=$(submit_dependent_job "scripts/learned_formula_enhanced_hpc.sh" "Enhanced Learned Formula" $degree_job)

echo ""
echo "Phase 4: Summary Analysis (depends on all previous phases)"
echo "========================================================"

# Create summary script
cat > scripts/summary_analysis.sh << 'EOL'
#!/bin/bash
#SBATCH --job-name=degree_summary
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=../logs/summary_analysis_%j.out
#SBATCH --error=../logs/summary_analysis_%j.err

# Summary analysis across all results
module purge
module load python/3.8.5
source ~/miniconda3/envs/hetionet/bin/activate

cd $SLURM_SUBMIT_DIR/..

echo "Running summary analysis..."

python3 << 'EOF'
import pandas as pd
from pathlib import Path
import json
import numpy as np

# Aggregate all results
results_dir = Path('results')
summary = {
    'degree_analysis': {},
    'enhanced_model_testing': {},
    'learned_formula': {},
    'overall_stats': {}
}

# Count generated files
def count_files(directory, pattern='*'):
    if directory.exists():
        return len(list(directory.glob(pattern)))
    return 0

# Degree analysis summary
degree_dir = results_dir / 'degree_analysis_hpc'
if degree_dir.exists():
    csv_files = count_files(degree_dir, '*.csv')
    png_files = count_files(degree_dir, '*.png')
    summary['degree_analysis'] = {
        'csv_files': csv_files,
        'png_files': png_files,
        'total_files': csv_files + png_files
    }

# Enhanced model testing summary
model_dir = results_dir / 'enhanced_model_testing'
if model_dir.exists():
    notebook_files = count_files(model_dir, '*.ipynb')
    summary['enhanced_model_testing'] = {
        'notebook_files': notebook_files
    }

# Learned formula summary
formula_dir = results_dir / 'learned_analytical_enhanced_hpc'
if formula_dir.exists():
    total_subdirs = len([d for d in formula_dir.iterdir() if d.is_dir()])
    notebook_files = count_files(formula_dir, '**/*.ipynb')
    summary['learned_formula'] = {
        'subdirectories': total_subdirs,
        'notebook_files': notebook_files
    }

# Overall statistics
total_files = (
    summary.get('degree_analysis', {}).get('total_files', 0) +
    summary.get('enhanced_model_testing', {}).get('notebook_files', 0) +
    summary.get('learned_formula', {}).get('notebook_files', 0)
)

summary['overall_stats'] = {
    'total_output_files': total_files,
    'analysis_complete': total_files > 0
}

# Save summary
summary_file = results_dir / 'degree_analysis_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Analysis Summary:")
print(f"================")
print(f"Degree Analysis Files: {summary.get('degree_analysis', {}).get('total_files', 0)}")
print(f"Model Testing Files: {summary.get('enhanced_model_testing', {}).get('notebook_files', 0)}")
print(f"Learned Formula Files: {summary.get('learned_formula', {}).get('notebook_files', 0)}")
print(f"Total Output Files: {total_files}")
print(f"Analysis Complete: {summary['overall_stats']['analysis_complete']}")
print(f"Summary saved to: {summary_file}")
EOF

echo "Summary analysis completed!"
EOL

chmod +x scripts/summary_analysis.sh

# Submit summary job with dependencies
dependency_list="${degree_job}:${model_job}:${formula_job}"
summary_job=$(submit_dependent_job "scripts/summary_analysis.sh" "Summary Analysis" $dependency_list)

echo ""
echo "========================================================"
echo "ALL JOBS SUBMITTED SUCCESSFULLY"
echo "========================================================"
echo "Job Dependencies:"
echo "  Phase 1 (Degree Analysis): $degree_job"
echo "  Phase 2 (Model Testing): $model_job (parallel)"
echo "  Phase 3 (Learned Formula): $formula_job (depends on $degree_job)"
echo "  Phase 4 (Summary): $summary_job (depends on all)"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/degree_analysis_${degree_job}_*.out"
echo ""
echo "Expected completion time: ~8-12 hours"
echo "Total compute hours: ~6-8 hours × 24 edge types = 144-192 core-hours"
echo "========================================================"