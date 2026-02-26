# Phase 1: Oracle Ceiling Analysis - Quick Start

## What Phase 1 Does

Computes **theoretical maximum performance (oracle)** for pathway prediction across all 119 2-hop metapaths in Hetionet.

**Key outputs**:
- Oracle r (exact degrees): Theoretical ceiling
- Oracle r (binned 10×10): Ceiling for binned approach
- Gap (exact - binned): Information loss from binning

**This tells us**: Maximum achievable r before investing in model development.

## Fixed Issues from HPC Run

1. **ConstantInputWarning**: Now handles metapaths with constant pathway counts (returns NaN)
2. **SimpleNN TypeError**: Removed feature/binning tests (require trained models, moved to Phase 2)

**Phase 1 now only computes oracle** - fast, model-free ceiling analysis.

## How to Run on HPC

### Step 1: Submit Array Job

```bash
# From repository root
sbatch scripts/phase1_ceiling_analysis_array.sh
```

**Job details**:
- 119 tasks (one per metapath)
- 2 hours per task
- 16GB memory per task
- Parallel execution (~1 minute wall time)

### Step 2: Monitor Progress

```bash
# Check job status
squeue -u $USER | grep ceiling

# Count completed
ls results/ceiling_analysis/*/summary.txt | wc -l
# Should reach 119

# Check for failures
grep -l ERROR logs/ceiling_*.err
```

### Step 3: Summarize Results

```bash
# After all jobs complete
python summarize_ceiling_results.py
```

**Outputs**:
- `results/ceiling_analysis_summary.csv`
- `results/ceiling_analysis_summary.md`

## Expected Outputs Per Metapath

**`results/ceiling_analysis/{metapath}/`**:
- `summary.txt`: Key results
  ```
  Metapath: CtDaG
  Oracle (exact): r = 0.9823
  Oracle (binned): r = 0.9654
  Gap (exact - binned): 0.0169
  Conclusion: Close to ceiling (1-3% gap)
  ```
- `oracle_exact_r.txt`: Exact degree ceiling
- `oracle_binned_r.txt`: Binned degree ceiling
- `binning_resolution.csv`: Placeholder (for Phase 2)
- `feature_sufficiency.csv`: Placeholder (for Phase 2)
- `ceiling_analysis_results.pkl`: Full results object

## Interpreting Results

**Oracle r values**:
- **r > 0.95**: Excellent predictability
- **0.80 < r < 0.95**: Good predictability
- **r < 0.80**: Low predictability (may be uninformative metapath)
- **r = NaN**: Constant pathway counts (no variance to predict)

**Gap (exact - binned)**:
- **< 0.01**: Binning preserves almost all information
- **0.01-0.05**: Moderate information loss
- **> 0.05**: Significant information loss, consider finer bins

## What Comes Next

**After Phase 1**:

1. Review `results/ceiling_analysis_summary.md`
2. Identify metapaths with highest oracle r (most predictable)
3. Check gap distribution (is binning sufficient?)

**Decision gates**:
- If most metapaths have oracle_binned_r > 0.95: Proceed to Phase 4 (long paths)
- If oracle_binned_r in 0.85-0.95 range: Proceed to Phase 2 (steelman baselines)
- If gap > 0.05 common: Test finer binning before Phase 2

## Troubleshooting

**"Metapath list not found"**:
```bash
python src/metapath_utils.py
# Regenerates data/2hop_metapaths.txt
```

**"No pathways found"**:
- Some metapaths may have zero pathways in Hetionet
- This is expected, job will exit gracefully

**"Permutation XXX not found"**:
```bash
ls data/permutations/*.hetmat
# Should show 000.hetmat through 019.hetmat
```

**Memory errors on specific metapaths**:
- Some metapaths (e.g., GcG->GiG) have millions of pathways
- Increase `--mem=32G` in SLURM script for resubmission

## Files Created

**Scripts**:
- `src/metapath_utils.py` - Enumerate 2-hop metapaths (119 total)
- `src/ceiling_analysis.py` - Oracle computation
- `run_phase1_ceiling_single_metapath.py` - Single metapath runner
- `summarize_ceiling_results.py` - Aggregate results

**HPC**:
- `scripts/phase1_ceiling_analysis_array.sh` - SLURM array job
- `data/2hop_metapaths.txt` - List of metapaths for array

**Docs**:
- `docs/PHASE1_CEILING_ANALYSIS.md` - Detailed documentation
- `PHASE1_QUICKSTART.md` - This file

## Example: Run Single Metapath Locally (if possible)

```bash
# Test one metapath
python run_phase1_ceiling_single_metapath.py --metapath CtDaG

# Results in: results/ceiling_analysis/CtDaG/
```

**Note**: May not work locally if sklearn not installed. Designed for HPC.

## Timeline

- **Submission**: 30 seconds
- **Execution**: 1-2 minutes wall time (parallel)
- **Summarization**: 10 seconds
- **Total**: < 5 minutes from submission to results

## Contact

Questions? See:
- Full docs: `docs/PHASE1_CEILING_ANALYSIS.md`
- Overall plan: `docs/HONEST_BENCHMARK_COMPARISON.md`
