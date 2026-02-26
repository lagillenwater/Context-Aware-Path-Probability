## Phase 1: Ceiling Analysis

Determines theoretical maximum performance for pathway prediction using degree-only features.

### Overview

**Goal**: Establish performance ceiling before investing in model development.

**Data**:
- Training: Permutations 000-014 (15 permutations)
- Testing: Permutations 015-019 (5 permutations)
- Metapaths: All 119 valid 2-hop metapaths in Hetionet

**Analyses**:
1. Oracle upper bound (exact and binned degrees)
2. Binning resolution test (5x5, 10x10, 20x20, 50x50)
3. Feature sufficiency test (sets A-F, 102-116 features)

### Quick Start

**Generate metapath list** (done once):
```bash
python src/metapath_utils.py
# Creates data/2hop_metapaths.txt with 119 metapaths
```

**Test single metapath locally**:
```bash
python run_phase1_ceiling_single_metapath.py --metapath CtDaG
# Results: results/ceiling_analysis/CtDaG/
```

**Run all 119 metapaths on HPC**:
```bash
# Submit array job (119 tasks)
sbatch scripts/phase1_ceiling_analysis_array.sh

# Monitor progress
squeue -u $USER

# Check results
ls results/ceiling_analysis/
```

**Summarize results**:
```bash
python summarize_ceiling_results.py
# Creates results/ceiling_analysis_summary.csv
# Creates results/ceiling_analysis_summary.md
```

### Feature Sets Tested

| Set | Features | Dimensionality | Description |
|-----|----------|----------------|-------------|
| A   | Baseline | 102 | 2 degree bins + 100-dim histogram |
| B   | + log transforms | 104 | A + log(source_deg), log(target_deg) |
| C   | + summary stats | 109 | B + mean, std, min, max, median of intermediates |
| D   | + neighbor context | 111 | C + 2nd-order degree statistics |
| E   | + polynomial | 113 | D + source_deg², target_deg² |
| F   | + interactions | 116 | E + degree products |

### Outputs per Metapath

**`results/ceiling_analysis/{metapath}/`**:
- `oracle_exact_r.txt`: Ceiling with exact degrees
- `oracle_binned_r.txt`: Ceiling with 10x10 bins
- `binning_resolution.csv`: Performance vs bin count
- `binning_resolution.png`: Plot of r vs bins
- `feature_sufficiency.csv`: Performance vs feature set
- `feature_sufficiency.png`: Bar chart of r by set
- `ceiling_analysis_results.pkl`: Full results object
- `summary.txt`: Key findings

### Interpretation

**Oracle Gap Analysis**:
- `gap < 0.01`: Near-optimal (within 1% of ceiling)
- `0.01 ≤ gap < 0.03`: Close to ceiling
- `gap ≥ 0.03`: Significant headroom for improvement

**Binning Resolution**:
- Plateaus early → fundamental limit
- Keeps increasing → need finer bins

**Feature Sufficiency**:
- Identifies minimal feature set
- SHAP analysis shows which features matter most

### Computational Cost

**Per metapath**:
- Oracle: ~0.2 seconds
- Binning test (4 resolutions): ~20 seconds
- Feature test (6 sets): ~35 seconds
- **Total: ~1 minute per metapath**

**Full analysis** (119 metapaths):
- Sequential: ~2 hours
- Parallel (HPC array): ~1 minute wall time

**Memory**: 16GB per task (conservative)

### Example Metapaths

**Pharmacological**:
- CtDaG: Compound-treats-Disease-associates-Gene
- CpDaG: Compound-palliates-Disease-associates-Gene
- CbGiG: Compound-binds-Gene-interacts-Gene

**Disease**:
- DrDaG: Disease-resembles-Disease-associates-Gene
- DlApD: Disease-localizes-Anatomy-participates-Disease
- DpSpD: Disease-presents-Symptom-presents-Disease

**Gene**:
- GcGiG: Gene-covaries-Gene-interacts-Gene
- GrGpBP: Gene-regulates-Gene-participates-Biological Process
- GiGpPW: Gene-interacts-Gene-participates-Pathway

### Decision Gates

**After Phase 1, decide**:

1. If oracle_binned_r - best_model_r < 0.01 for most metapaths:
   - Current DegreeSignatureNN is near-optimal
   - Skip feature engineering (Phase 3)
   - Proceed directly to long path validation (Phase 4)

2. If oracle_binned_r - best_model_r > 0.05:
   - Significant headroom exists
   - Proceed to Phase 2 (steelman baselines)
   - Invest in Phase 3 (feature engineering)

3. If oracle_exact_r - oracle_binned_r > 0.05:
   - Binning loses significant information
   - Test finer bins (20x20, 50x50)
   - Consider bin-free approaches

### HPC Job Management

**Check job status**:
```bash
squeue -u $USER | grep ceiling
```

**Check failed jobs**:
```bash
grep -l ERROR logs/ceiling_*.err
```

**Resubmit failed tasks**:
```bash
# Get failed task IDs
failed_tasks=$(grep -l ERROR logs/ceiling_*.err | sed 's/.*_//;s/.err//')

# Resubmit specific tasks
sbatch --array=$failed_tasks scripts/phase1_ceiling_analysis_array.sh
```

**Monitor progress**:
```bash
# Count completed summaries
ls results/ceiling_analysis/*/summary.txt | wc -l

# Expected: 119
```

### Next Steps

After Phase 1 completes:

1. Run summary: `python summarize_ceiling_results.py`
2. Review `results/ceiling_analysis_summary.md`
3. Identify metapaths with large oracle gaps
4. Decide whether to proceed with Phase 2 or skip to Phase 4

### Files Created

**Scripts**:
- `src/metapath_utils.py`: Enumerate 2-hop metapaths
- `src/enhanced_features.py`: Extract 116-dim features
- `src/ceiling_analysis.py`: Oracle and feature tests
- `run_phase1_ceiling_single_metapath.py`: Single metapath runner
- `summarize_ceiling_results.py`: Aggregate results

**HPC**:
- `scripts/phase1_ceiling_analysis_array.sh`: SLURM array job
- `data/2hop_metapaths.txt`: List of 119 metapaths

**Documentation**:
- `docs/PHASE1_CEILING_ANALYSIS.md`: This file

### Troubleshooting

**"Metapath list not found"**:
```bash
python src/metapath_utils.py
```

**"No pathways found for this metapath"**:
- Metapath may have zero paths in Hetionet
- Check with: `cat data/2hop_metapaths.txt | grep {metapath}`

**"Permutation not found"**:
- Verify permutations exist: `ls data/permutations/*.hetmat`
- Should see 000.hetmat through 019.hetmat

**Memory errors**:
- Increase `--mem` in SLURM script
- Some metapaths (e.g., GcG→GiG) have many pathways

### Contact

For questions about this analysis, see:
- `docs/HONEST_BENCHMARK_COMPARISON.md`: Overall approach
- `docs/PERFORMANCE_CEILING.md`: Results (generated after Phase 1)
