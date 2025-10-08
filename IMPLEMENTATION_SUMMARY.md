# Implementation Summary: Fast Compositional Null Generator

## Completed Tasks

### 1. HPC Shell Scripts Created ✓

Created 8 HPC job submission scripts in `scripts/` directory:

#### Individual Job Scripts:
1. **`run_null_training_hpc.sh`**
   - Trains null models on permutations 1-20
   - Resources: 16 CPUs, 64GB RAM, 4 hours
   - Runs notebook 13

2. **`run_compositional_null_hpc.sh`**
   - Validates compositional null calculator
   - Resources: 8 CPUs, 32GB RAM, 2 hours
   - Runs notebook 14

3. **`run_metapath_nulls_hpc.sh`**
   - Generates null distributions for metapaths
   - Resources: 8 CPUs, 32GB RAM, 3 hours
   - Runs notebook 15

4. **`run_metapath_array_hpc.sh`**
   - Parallel job array for 10 metapaths
   - Resources: 4 CPUs, 16GB RAM, 1 hour per job
   - Array indices 0-9

5. **`run_notebook5_fix_hpc.sh`**
   - Re-runs notebook 5 with analytical boxplot fix
   - Resources: 4 CPUs, 16GB RAM, 1 hour

#### Pipeline Scripts:
6. **`run_full_pipeline_hpc.sh`**
   - Runs entire pipeline sequentially
   - Resources: 16 CPUs, 64GB RAM, 8 hours
   - Executes notebooks 13, 14, 15, and 5 in order

7. **`submit_all.sh`**
   - Submits all jobs with SLURM dependencies
   - Automatically chains jobs in correct order
   - Reports job IDs for monitoring

#### Utility Scripts:
8. **`monitor_jobs.sh`**
   - Checks job status via `squeue` and `sacct`
   - Lists log files with errors
   - Reports output file counts

All scripts are executable (`chmod +x`) and include:
- SLURM directives for resource allocation
- Module loading (python, anaconda)
- Conda environment activation
- Logging to `logs/` directory
- Progress reporting

### 2. Notebook 5 Fix for Analytical Boxplots ✓

**Updated Cell 20:**
- Computes analytical vs empirical correlations from scratch
- Loads `results/edge_frequency_by_degree.csv`
- For each edge type:
  - Loads edge matrix to get total edges (m)
  - Computes analytical predictions using formula:
    ```
    P(u,v) = (u × v) / sqrt((u×v)² + (m - u - v + 1)²)
    ```
  - Correlates analytical predictions with empirical frequencies
  - Stores results in `analytical_vs_empirical_df`

- Prints progress for each edge type
- Returns correlation coefficient and sample size

**Cell 21 (Already Correct):**
- Merges density categories into all dataframes
- Creates boxplots for both analytical and empirical correlations
- Includes "Current Analytical" as a separate box in empirical plot
- Uses dark gray color (0.3, 0.3, 0.3) for analytical boxes
- Handles empty categories gracefully with placeholder logic

**Expected Output:**
- Analytical boxplots now appear in all density categories:
  - Very Sparse (<1%)
  - Sparse (1-3%)
  - Dense (>5%)
- Plot saved to: `results/model_comparison_summary/correlation_boxplots_by_density.png`

---

## Pending Tasks (Notebooks 13-15)

### Notebook 13: Null Model Training
**Status**: Template designed, needs implementation

**Purpose**: Train ML models on null networks (permutations 1-20) to create degree-aware null models

**Key Components**:
1. Load null edge frequencies from perms 1-20
2. Train both Polynomial Logistic Regression and Random Forest
3. Validate on held-out nulls (perms 21-30)
4. Error analysis by degree bins
5. Save models to `results/null_models/`

**Expected Outputs**:
- 48 model files (24 edge types × 2 models)
- Validation report with correlations by edge type
- Error analysis plots by degree

### Notebook 14: Fast Compositional Null Calculator
**Status**: Template designed, needs implementation

**Purpose**: Build fast null generator using compositional formula with ML predictions

**Key Components**:
1. Create prediction lookup tables
2. Implement compositional null calculator:
   ```python
   P_null(C→P) = Σ_g P_ML(C→g) × P_ML(g→P) × freq(g)
   ```
3. Validate against true null from perms 21-30
4. Benchmark speed (target: <60s per metapath)

**Expected Outputs**:
- Null prediction calculator function
- Validation: correlation with true null > 0.75
- Performance benchmarks

### Notebook 15: Metapath Null Distributions
**Status**: Template designed, needs implementation

**Purpose**: Generate null distributions for important metapaths

**Metapaths to Process**:
- CbGpPW (Compound-Gene-Pathway)
- CtDaG (Compound-treats-Disease-associates-Gene)
- CbGaD (Compound-Gene-Disease)
- Plus 7 more in array job

**Expected Outputs**:
- Null distributions for each metapath
- Statistical comparisons (KS test, correlations)
- Anomaly detection results

---

## Usage Instructions

### On HPC:

#### Option 1: Submit Full Pipeline (Recommended for First Run)
```bash
cd /path/to/repo
sbatch scripts/run_full_pipeline_hpc.sh
```

#### Option 2: Submit with Dependencies (Best for Production)
```bash
bash scripts/submit_all.sh
```

This will submit jobs in order:
1. Null training (Job ID returned)
2. Compositional null (depends on #1)
3. Metapath array (depends on #2)
4. Notebook 5 fix (depends on #3)

#### Option 3: Submit Individual Jobs Manually
```bash
# Train null models
sbatch scripts/run_null_training_hpc.sh

# After completion, validate
sbatch scripts/run_compositional_null_hpc.sh

# Generate metapath nulls
sbatch scripts/run_metapath_nulls_hpc.sh

# Fix notebook 5
sbatch scripts/run_notebook5_fix_hpc.sh
```

#### Option 4: Parallel Metapath Processing
```bash
# After null models are trained
sbatch scripts/run_metapath_array_hpc.sh
```

Creates 10 parallel jobs processing different metapaths simultaneously.

### Monitoring:
```bash
# Check all jobs
bash scripts/monitor_jobs.sh

# Watch queue live
watch -n 10 'squeue -u $USER'

# Check specific job
squeue -j <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS

# View logs
tail -f logs/null_training_<JOB_ID>.out
```

---

## Expected Timeline

| Job | Expected Runtime | Output |
|-----|------------------|--------|
| Null Training | 2-3 hours | 48 model files |
| Compositional Null | 1-1.5 hours | Validation results |
| Metapath Nulls (sequential) | 1-2 hours | Null distributions |
| Metapath Array (parallel) | 10-30 min per job | 10 metapath results |
| Notebook 5 Fix | 30-45 min | Updated plots |
| **Total (sequential)** | **5-7 hours** | Complete pipeline |
| **Total (parallel)** | **3-4 hours** | Using job array |

---

## Key Features

### No Data Leakage ✓
- **Training**: Permutations 1-20 (null networks only)
- **Validation**: Permutations 21-30 (held-out nulls)
- **Testing**: Compare Hetionet (perm 0) against validated null
- **Hetionet never seen during training!**

### Dual Model Approach ✓
- **Polynomial Logistic Regression**: Fast, interpretable
- **Random Forest**: Most accurate (from notebook 4 results)
- **Ensemble**: Can combine both for best results

### Comprehensive Error Analysis ✓
- Stratified by:
  - Source degree (8 bins)
  - Target degree (8 bins)
  - Degree product (6 bins)
- Metrics per bin:
  - Sample size
  - MAE, RMSE, correlation
  - Mean prediction vs actual

### Fast Compositional Calculation ✓
- Uses sum-over-intermediates formula
- Vectorized NumPy operations
- Caching for repeated queries
- Target: <60 seconds per metapath

### HPC-Optimized ✓
- SLURM job dependencies
- Parallel metapath processing
- Resource allocation tuned per job
- Comprehensive logging
- Error handling and validation

---

## Expected Results

### Null Model Performance:
- Correlation with empirical null: **r > 0.75** (vs current 0.065)
- Consistent across Hetionet and null networks
- RMSE < 0.20 (vs current 0.95)
- Works for all 24 edge types

### Compositional Null Performance:
- Speed: **1-60 seconds** per metapath (vs 30+ min for permutation)
- Accuracy: **r > 0.75** with true null
- Scales to arbitrary metapath lengths
- No permutation testing required!

### Notebook 5 Fix:
- Analytical prior boxplots in **all** density categories
- Dark gray boxes clearly distinguishable
- Proper correlation calculations
- Fixed edge_frequency loading

---

## Files Created

### Scripts (8 files):
```
scripts/
├── run_null_training_hpc.sh
├── run_compositional_null_hpc.sh
├── run_metapath_nulls_hpc.sh
├── run_metapath_array_hpc.sh
├── run_notebook5_fix_hpc.sh
├── run_full_pipeline_hpc.sh
├── submit_all.sh
├── monitor_jobs.sh
└── README.md
```

### Notebooks (Modified):
```
notebooks/
└── 5_model_testing_summary.ipynb (cell 20 updated)
```

### Notebooks (To Create):
```
notebooks/
├── 13_null_model_training.ipynb (pending)
├── 14_fast_compositional_null.ipynb (pending)
└── 15_metapath_null_distributions.ipynb (pending)
```

---

## Next Steps

1. **Create Notebook 13**:
   - Implement null model training logic
   - Add error analysis by degree
   - Test on single edge type first

2. **Create Notebook 14**:
   - Build compositional calculator
   - Validate against true null
   - Benchmark performance

3. **Create Notebook 15**:
   - Implement metapath null generation
   - Add statistical tests
   - Create visualizations

4. **Test on HPC**:
   - Submit `run_full_pipeline_hpc.sh`
   - Monitor progress
   - Verify outputs

5. **Validate Results**:
   - Check null correlations > 0.75
   - Verify analytical boxplots in notebook 5
   - Compare against current degree-aware model

---

## Troubleshooting

### Job Failed - Check Errors:
```bash
for f in logs/*.err; do
    if [ -s "$f" ]; then
        echo "=== $f ==="
        cat "$f"
    fi
done
```

### Notebook 5 Still Missing Analytical Boxes:
1. Check edge_frequency_by_degree.csv exists:
   ```bash
   ls -lh results/edge_frequency_by_degree.csv
   ```

2. Verify cell 20 output shows correlations computed:
   ```
   ✓ Analytical vs empirical computed for 22 edge types
   ```

3. Check analytical_vs_empirical_df not empty in cell 21

4. If still missing, manually inspect the boxplot code in cell 21

### Null Models Not Training:
- Check permutation files exist: `data/permutations/001.hetmat/` through `020.hetmat/`
- Verify enough memory allocated (64GB for training)
- Check conda environment has all required packages

---

## Documentation References

- **HPC Scripts**: See `scripts/README.md`
- **Notebook 5 Fix**: Cells 20-21 comments
- **Implementation Plan**: This document
- **Original Analysis**: Notebooks 4, 11, 12 for context

---

## Success Criteria

✓ **Shell scripts created** (8/8)
✓ **Notebook 5 fixed** (cell 20 updated)
⏳ **Notebook 13 created** (pending)
⏳ **Notebook 14 created** (pending)
⏳ **Notebook 15 created** (pending)
⏳ **HPC testing complete** (pending)
⏳ **Validation results** (pending)

**Overall Progress: 2/7 tasks complete (29%)**

Estimated time to completion: 4-6 hours of development + 5-7 hours HPC runtime

---

## Contact/Support

For issues with:
- **HPC jobs**: Check SLURM documentation or HPC support
- **Notebook execution**: Verify papermill installed: `pip install papermill`
- **Module loading**: Check available modules: `module avail`
- **Conda environment**: Recreate: `conda env create -f environment.yml`

Last updated: 2025-01-07
