# HPC Job Scripts for Null Model Pipeline

This directory contains SLURM batch scripts for running the null model pipeline on an HPC cluster.

## Scripts Overview

### Individual Component Scripts

1. **`run_null_training_hpc.sh`**
   - Trains null models on permutations 1-20
   - Resources: 16 CPUs, 64GB RAM, 4 hours
   - Output: 48 model files in `results/null_models/`

2. **`run_compositional_null_hpc.sh`**
   - Validates compositional null on permutations 21-30
   - Resources: 8 CPUs, 32GB RAM, 2 hours
   - Output: Validation results in `results/compositional_null/`

3. **`run_metapath_nulls_hpc.sh`**
   - Generates null distributions for metapaths
   - Resources: 8 CPUs, 32GB RAM, 3 hours
   - Output: Null distributions in `results/metapath_nulls/`

4. **`run_metapath_array_hpc.sh`**
   - Parallel processing of 10 metapaths using job arrays
   - Resources: 4 CPUs, 16GB RAM, 1 hour per metapath
   - Output: Individual metapath results

5. **`run_notebook5_fix_hpc.sh`**
   - Re-runs notebook 5 with analytical boxplot fix
   - Resources: 4 CPUs, 16GB RAM, 1 hour
   - Output: Updated summary plots

### Pipeline Scripts

6. **`run_full_pipeline_hpc.sh`**
   - Runs entire pipeline sequentially (steps 1-4)
   - Resources: 16 CPUs, 64GB RAM, 8 hours
   - Use this for a complete end-to-end run

7. **`submit_all.sh`**
   - Submits individual jobs with dependencies
   - Jobs run sequentially, each starting after previous completes
   - Recommended for production runs

### Utility Scripts

8. **`monitor_jobs.sh`**
   - Checks job status, logs, and output files
   - Run periodically to monitor progress

## Usage

### Quick Start (Submit with Dependencies)

```bash
cd /path/to/repo
bash scripts/submit_all.sh
```

This submits all jobs with proper dependencies and reports job IDs.

### Submit Full Pipeline (Sequential)

```bash
sbatch scripts/run_full_pipeline_hpc.sh
```

### Submit Individual Jobs Manually

```bash
# Step 1: Train null models
JOB1=$(sbatch --parsable scripts/run_null_training_hpc.sh)

# Step 2: Validate compositional null (after Step 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/run_compositional_null_hpc.sh)

# Step 3: Generate metapath nulls (after Step 2)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 scripts/run_metapath_nulls_hpc.sh)

# Step 4: Fix notebook 5 (after Step 3)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 scripts/run_notebook5_fix_hpc.sh)
```

### Submit Parallel Metapath Jobs

```bash
# Requires null models to be trained first
sbatch scripts/run_metapath_array_hpc.sh
```

This creates 10 parallel jobs (array indices 0-9), one per metapath.

## Monitoring

### Check Job Status

```bash
bash scripts/monitor_jobs.sh
```

### Watch Queue Live

```bash
watch -n 10 'squeue -u $USER'
```

### Check Specific Job

```bash
squeue -j <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS
```

### View Logs

```bash
# Output logs
ls -lth logs/*.out | head

# Error logs
ls -lth logs/*.err | head

# Tail a running job
tail -f logs/null_training_<JOB_ID>.out
```

## Output Files

### Expected Directory Structure

```
results/
├── null_models/
│   ├── CbG_poly_null.pkl
│   ├── CbG_poly_features.pkl
│   ├── CbG_rf_null.pkl
│   ├── ... (48 model files total)
│   └── validation_results.csv
├── compositional_null/
│   ├── validation_metrics.csv
│   ├── error_analysis_by_degree.csv
│   └── plots/
├── metapath_nulls/
│   ├── CbGpPW_null_distribution.csv
│   ├── CtDaG_null_distribution.csv
│   └── ...
└── model_comparison_summary/
    └── correlation_boxplots_by_density.png (with analytical boxes!)

notebooks/executed/
├── 13_null_model_training_executed.ipynb
├── 14_fast_compositional_null_executed.ipynb
├── 15_metapath_null_distributions_executed.ipynb
└── 5_model_testing_summary_executed.ipynb

logs/
├── null_training_<JOB_ID>.out
├── null_training_<JOB_ID>.err
└── ...
```

## Expected Runtimes

| Job | CPUs | Memory | Time Limit | Expected Runtime |
|-----|------|--------|------------|------------------|
| Null Training | 16 | 64GB | 4 hours | 2-3 hours |
| Compositional Null | 8 | 32GB | 2 hours | 1-1.5 hours |
| Metapath Nulls | 8 | 32GB | 3 hours | 1-2 hours |
| Metapath Array (each) | 4 | 16GB | 1 hour | 10-30 min |
| Notebook 5 Fix | 4 | 16GB | 1 hour | 30-45 min |
| **Full Pipeline** | 16 | 64GB | 8 hours | 5-7 hours |

## Troubleshooting

### Job Failed - Check Errors

```bash
# Find error logs with content
for f in logs/*.err; do
    if [ -s "$f" ]; then
        echo "=== $f ==="
        cat "$f"
    fi
done
```

### Job Out of Memory

Increase `--mem` in the SBATCH header:
```bash
#SBATCH --mem=128GB  # Double the memory
```

### Job Timeout

Increase `--time` in the SBATCH header:
```bash
#SBATCH --time=08:00:00  # 8 hours instead of 4
```

### Missing Conda Environment

```bash
# On HPC login node, create environment
conda create -n hetionet_env python=3.9
conda activate hetionet_env
pip install -r requirements.txt
```

### Module Not Found

Check which modules are available:
```bash
module avail python
module avail anaconda
```

Update the `module load` lines in scripts accordingly.

## Customization

### Change Partition

Edit the `#SBATCH --partition=` line:
```bash
#SBATCH --partition=short    # < 4 hours
#SBATCH --partition=long     # > 4 hours
#SBATCH --partition=gpu      # GPU nodes
```

### Change Resource Allocation

```bash
#SBATCH --cpus-per-task=32   # More CPUs
#SBATCH --mem=128GB          # More memory
#SBATCH --time=12:00:00      # More time
```

### Add Email Notifications

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.edu
```

## Notes

- All scripts assume `papermill` is installed for notebook execution
- Scripts create necessary directories automatically
- Logs are timestamped with job ID for easy tracking
- Job dependencies ensure correct execution order
- Array jobs enable parallel processing of metapaths
