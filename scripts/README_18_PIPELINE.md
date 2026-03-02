# Degree Signature Pipeline Usage

## Quick Start

From the scripts directory on HPC:

```bash
cd /projects/$USER/repositories/Context-Aware-Path-Probability/scripts

# Option 1: Submit entire pipeline with dependencies (RECOMMENDED)
bash 18_submit_all.sh

# Option 2: Submit jobs individually
sbatch 17b_compositional_failure_analysis.sh
sbatch 18a_data_preparation.sh
sbatch 18b_train_random.sh
sbatch 18c_train_degree_product.sh
sbatch 18d_train_negbin_glm.sh
sbatch 18e_train_random_forest.sh
sbatch 18f_train_degree_signature_nn.sh
sbatch 18g_validate_cv_all_models.sh
sbatch 18i_benchmark_summary.sh
```

## What `18_submit_all.sh` Does

The master script:
1. Creates necessary directories
2. Submits jobs using `sbatch --parsable` to get job IDs
3. Sets up SLURM dependencies (--dependency=afterok:$JOB_ID)
4. Tracks and displays all job IDs
5. Provides monitoring commands

**It does NOT run the jobs directly** - it just submits them to SLURM.

## Pipeline Structure

```
17b Baseline (6h)
  ↓ (after 17b completes)
18a Data Prep (8 tasks × 6h)
  ↓ (after 18a completes)
18b-f Training (5 models, parallel)
  ├─ 18b Random (8 tasks × 0.5h)
  ├─ 18c Degree Product (8 tasks × 0.5h)
  ├─ 18d NegBin GLM (8 tasks × 1h)
  ├─ 18e Random Forest (8 tasks × 2h)
  └─ 18f Degree Sig NN (8 tasks × 2h)
  ↓ (after all training completes)
18g CV Validation (8 tasks × 1h)
  ↓ (after 18g completes)
18i Benchmark Summary (single job, 0.5h)
```

## Individual Job Submission

If you prefer to submit jobs one at a time:

```bash
# Submit 17b baseline
sbatch 17b_compositional_failure_analysis.sh
# Wait for completion, then:

# Submit data prep
sbatch 18a_data_preparation.sh
# Wait for all 8 array tasks, then:

# Submit training (all can run in parallel)
sbatch 18b_train_random.sh
sbatch 18c_train_degree_product.sh
sbatch 18d_train_negbin_glm.sh
sbatch 18e_train_random_forest.sh
sbatch 18f_train_degree_signature_nn.sh
# Wait for all training, then:

# Submit validation
sbatch 18g_validate_cv_all_models.sh
# Wait for validation, then:

# Submit summary
sbatch 18i_benchmark_summary.sh
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# Check array job
squeue -j <JOB_ID>_*

# View job details
scontrol show job <JOB_ID>

# Check logs
tail -f ../logs/18a_data_prep_<JOB_ID>_<ARRAY_ID>.out
tail -f ../logs/18a_data_prep_<JOB_ID>_<ARRAY_ID>.err
```

## Canceling Jobs

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel array job
scancel <JOB_ID>

# Cancel specific array task
scancel <JOB_ID>_<ARRAY_ID>

# Cancel all your jobs
scancel -u $USER
```

## Resource Allocation

| Job | Memory | Time | Array | Total Cost |
|-----|--------|------|-------|------------|
| 17b | 32GB | 6h | - | 6 node-hours |
| 18a | 64GB | 6h | 8 | 48 node-hours |
| 18b | 16GB | 0.5h | 8 | 2 node-hours |
| 18c | 16GB | 0.5h | 8 | 2 node-hours |
| 18d | 16GB | 1h | 8 | 4 node-hours |
| 18e | 32GB | 2h | 8 | 16 node-hours |
| 18f | 32GB | 2h | 8 | 16 node-hours |
| 18g | 16GB | 1h | 8 | 8 node-hours |
| 18i | 16GB | 0.5h | - | 0.5 node-hours |
| **TOTAL** | | | | **102.5 node-hours** |

Wall clock time: ~15-17 hours with dependencies

## Expected Outputs

```
results/pathway_nn/
├── training_data/
│   └── {metapath}_training_data.csv (8 files)
├── trained_models/
│   ├── {metapath}_Random.pkl (8 files)
│   ├── {metapath}_Degree_Product.pkl (8 files)
│   ├── {metapath}_NegBin_GLM.pkl (8 files)
│   ├── {metapath}_Random_Forest.pkl (8 files)
│   └── {metapath}_Degree_Sig_NN.pt (8 files)
├── benchmarks/
│   └── {metapath}_{model}_benchmark.json (40 files)
├── validation/
│   └── cv_results_{metapath}.csv (8 files)
└── figures/
    ├── efficiency_frontier.png
    ├── model_comparison_heatmap.png
    └── cv_performance_by_metapath.png
```

## Troubleshooting

### Job fails immediately
```bash
# Check error log
cat ../logs/<job_name>_<JOB_ID>_<ARRAY_ID>.err

# Common issues:
# - Missing conda environment: conda activate CAPP
# - Missing data files: check data/ directory
# - Memory exceeded: increase --mem in script
```

### Job queued but not running
```bash
# Check queue
squeue -u $USER

# Check job reason
squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Common reasons:
# - Priority: wait for higher priority jobs
# - Resources: requested resources not available
# - Dependency: waiting for another job
```

### Array job partially fails
```bash
# Check which tasks failed
sacct -j <JOB_ID> --format=JobID,State,ExitCode

# Resubmit specific tasks
sbatch --array=<TASK_ID> <script>.sh
```

## Notes

- All scripts use `--parsable` with sbatch to capture job IDs
- Dependencies use `--dependency=afterok:$JOB_ID` 
- Array indices start at 1 (SLURM convention)
- Logs go to `../logs/` (repository root)
- All paths relative to scripts/ directory
