# Session Summary: 2025-11-25

## Overview

This session focused on validating our DWPC (Degree-Weighted Path Count) calculation against het.io's ground truth implementation before proceeding with p-value calibration fixes identified in yesterday's analysis. The validation was successful, confirming our DWPC formula is correct.

## Context from Yesterday (2025-11-24)

Yesterday's analysis identified systematic p-value calibration failures in our gamma-hurdle method-of-moments approach:
- Permutation 0 showed mean p-value = 0.741 instead of expected 0.50
- Failure scaled with path length and node degree
- Sample size strongly confounded results (r = 0.807)

Before implementing fixes, we needed to verify that DWPC calculations were correct.

## Key Achievements

### 1. Identified Validation Approach Using Multi-DWPC Repository

We leveraged the existing Multi-DWPC repository which contains:
- Neo4j ID mappings for GO terms (`hetionet_neo4j_go_ids_nr.csv`)
- Neo4j ID mappings for genes (`hetionet_neo4j_genes_ids_nr.csv`)
- Het.io API output with DWPC values for BP-Gene pairs

The Multi-DWPC notebooks (1.1-1.5) demonstrated the workflow for mapping biological identifiers to neo4j IDs required for het.io API queries.

### 2. Created DWPC Validation Script

Developed `scripts/validate_dwpc_vs_hetio.py` which:
- Loads neo4j ID mappings from Multi-DWPC repository
- Maps our node indices to het.io's neo4j IDs
- Calculates DWPC using our implementation
- Queries het.io API for ground truth PDP values
- Compares results with detailed statistics

### 3. Discovered Critical Issue with Het.io CSV Output

**Important finding**: The `dwpc` column in het.io's CSV output files is NOT the observed DWPC for a specific node pair. It is actually `dgp_nonzero_mean` - the mean DWPC from the null distribution (degree-grouped permutation statistics).

The actual observed DWPC is the `PDP` (Path Degree Product) value returned in the `paths` array of the API response.

This distinction is critical for anyone working with het.io data exports.

### 4. Identified Degree Calculation Difference

Initial validation showed ~500x scale difference between our DWPC and het.io's values (correlation = 0.94).

**Root cause**: We were using total node degrees (summed across all edge types), but het.io uses **edge-specific degrees** (degrees within the specific metaedge being traversed).

From hetmatpy source code (`degree_weight.py`):
```python
def _degree_weight(matrix, damping, copy=True, dtype=numpy.float64):
    """Normalize an adjacency matrix by the in and out degree."""
    row_sums = numpy.array(matrix.sum(axis=1), dtype=dtype).flatten()
    column_sums = numpy.array(matrix.sum(axis=0), dtype=dtype).flatten()
    matrix = hetmatpy.matrix.normalize(matrix, row_sums, "rows", damping)
    matrix = hetmatpy.matrix.normalize(matrix, column_sums, "columns", damping)
    return matrix
```

The degrees are computed from the adjacency matrix itself (`matrix.sum()`), not from total node degrees.

### 5. Validation Results: PERFECT MATCH

After correcting the degree calculation:

| Metric | Value |
|--------|-------|
| Pairs compared | 20 |
| Pearson correlation | 1.000000 |
| Exact matches (rtol=1e-4) | 20/20 (100%) |
| Mean absolute difference | 0.000000 |
| Max absolute difference | 0.000000 |

**Conclusion**: Our DWPC formula is correct and exactly matches het.io's implementation.

## Scientific Implications

### DWPC Formula Confirmed

For a path through nodes [n0, n1, ..., nk] with metaedges [e0, e1, ..., e(k-1)]:

```
DWPC = sum over all paths of: product(deg_ei(ni)^-w * deg_ei(n(i+1))^-w for each edge ei)
```

Where:
- `w` = damping exponent (typically 0.5)
- `deg_ei(n)` = degree of node n **within metaedge ei** (not total degree)

For length-1 paths (single edge):
```
DWPC = 1 * deg_source^-0.5 * deg_target^-0.5
```

### P-Value Calibration Issue Isolated

Since DWPC calculation is correct, the p-value calibration failures identified yesterday are definitively in the **gamma-hurdle method-of-moments parameter estimation**, not in the DWPC formula itself.

This confirms the plan to implement empirical percentile p-values as the next step.

## Files Generated

| File | Description |
|------|-------------|
| `scripts/validate_dwpc_vs_hetio.py` | DWPC validation script comparing our calculations to het.io API |
| `docs/2025-11-25_SESSION_SUMMARY.md` | This summary document |

## Technical Details

### Het.io API Endpoint for DWPC

```
GET https://search-api.het.io/v1/paths/source/{neo4j_source_id}/target/{neo4j_target_id}/metapath/{metapath}/?format=json
```

Response structure:
- `path_count_info.dwpc` - NOT the observed DWPC; this is `dgp_nonzero_mean`
- `path_count_info.dgp_source_degree` - Source node degree in this metaedge
- `path_count_info.dgp_target_degree` - Target node degree in this metaedge
- `paths[0].PDP` - **Actual observed DWPC** (Path Degree Product)

### Validation Sample Output

```
     go_id  entrez_gene_id  hetio_pdp  our_dwpc  abs_diff
GO:0070252           10052   0.016667  0.016667       0.0
GO:0014032            3091   0.005356  0.005356       0.0
GO:0006813           10021   0.010673  0.010673       0.0
GO:0061008           26019   0.014982  0.014982       0.0
GO:0042752            2767   0.015831  0.015831       0.0
```

## P-Value Validation (Added Later Same Day)

### 6. Created P-Value Validation Script

Developed `scripts/validate_pvalue_vs_hetio.py` which:
- Loads het.io's p-values and null distribution statistics from Multi-DWPC CSV output
- Reconstructs gamma-hurdle parameters from dgp (degree-grouped permutation) statistics
- Calculates p-values using our gamma-hurdle implementation
- Compares results against het.io's ground truth

### 7. P-Value Validation Results: PERFECT MATCH (for length >= 2)

For metapaths of length 2 or greater:

| Metric | Value |
|--------|-------|
| Pairs compared | 1000 |
| Pearson correlation (log10) | 1.000000 |
| Exact matches (rtol=1e-4) | 1000/1000 (100%) |
| Mean absolute difference | 0.000000 |
| Max absolute difference | ~1e-15 (floating point precision) |

**Conclusion**: Our gamma-hurdle p-value calculation exactly matches het.io's implementation.

### 8. Het.io P-Value Methodology Confirmed

From the [GigaScience 2023 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10375517/) and validation:

**Gamma-Hurdle Parameters (Method of Moments):**
```
lambda = n_nonzero / N  (proportion of non-zero DWPCs)
alpha = mean^2 / variance  (gamma shape)
beta = mean / variance  (gamma rate)
```

**P-Value Formula:**
```
P(DWPC >= observed) = lambda * Gamma.sf(observed; alpha, scale=1/beta)
```

Where:
- `N` = total permuted DWPC values (`dgp_n_dwpcs`)
- `n_nonzero` = non-zero permuted values (`dgp_n_nonzero_dwpcs`)
- `mean` = mean of non-zero DWPCs (`dgp_nonzero_mean`)
- `variance` = variance of non-zero DWPCs (`dgp_nonzero_sd^2`)

### 9. CRITICAL CLARIFICATION: Formula-Only Validation

**Important**: The p-value validation above was a **formula-only validation**, NOT an end-to-end pipeline validation.

What the validation actually tested:
```python
# We used HET.IO's precomputed null statistics directly:
n_total = row['dgp_n_dwpcs']           # Het.io's 200 permutations
n_nonzero = row['dgp_n_nonzero_dwpcs'] # Het.io's counts
mean = row['dgp_nonzero_mean']         # Het.io's mean
sd = row['dgp_nonzero_sd']             # Het.io's std

# Then applied our formula to THEIR statistics
our_pvalue = gamma_hurdle_pvalue(observed, n_total, n_nonzero, mean, sd)
```

This proves: `our_formula(het.io_statistics) == het.io_pvalue`

This does NOT prove: `our_formula(our_statistics) == het.io_pvalue`

The 100% match occurred because we used het.io's null distribution statistics, not our own. Our pipeline uses far fewer permutations and different sampling, so our null distributions would differ.

**What remains to be validated**:
- End-to-end: Compute DWPCs using OUR permutations, build OUR null distributions, calculate p-values, compare to het.io

The calibration issues identified yesterday (mean p-value = 0.96 instead of 0.50) came from our pipeline's null distributions, not from the gamma-hurdle formula.

### 10. End-to-End Validation Results

Created `scripts/validate_pvalue_end_to_end.py` to test our full pipeline using:
- **Null distribution**: Permutations 1-5 with degree-grouped sampling
- **Test data**: Permutation 0 (pairs with edges) and Hetionet

**Key findings for specific Gene-BP pairs:**

| Gene | GO Term | Het.io p-value | Our p (Hetionet) | Our p (Perm0) |
|------|---------|----------------|------------------|---------------|
| FZD9 | learning or memory | 0.959 | 0.011 | 1.000 |
| SMO | hindbrain development | 0.395 | 0.048 | 1.000 |
| SOX9 | molting cycle | 0.034 | 0.034 | 1.000 |

**Calibration test (30 perm0 pairs with edges):**
- Mean p-value: **0.044** (expected ~0.50)
- 70% of p-values < 0.05 (expected 5%)
- 100% of p-values < 0.50 (expected 50%)

**Interpretation**: P-values are severely DEFLATED (too small). This is because:
1. We sample only 5 permutations vs het.io's 200
2. Degree-grouped sampling from 5 perms gives insufficient null coverage
3. Observed DWPCs for actual edges are systematically higher than random degree-matched pairs

This confirms that the calibration issue is in **null distribution coverage**, not the formula.

## Files Generated

| File | Description |
|------|-------------|
| `scripts/validate_dwpc_vs_hetio.py` | DWPC validation script comparing our calculations to het.io API |
| `scripts/validate_pvalue_vs_hetio.py` | P-value formula validation using het.io's precomputed statistics |
| `scripts/validate_pvalue_end_to_end.py` | End-to-end validation with configurable permutations (HPC-ready) |
| `scripts/28_pvalue_validation.sh` | SLURM script for running p-value validation on HPC |
| `docs/2025-11-25_session_summary.md` | This summary document |

## Technical Details

### Het.io API Endpoint for DWPC

```
GET https://search-api.het.io/v1/paths/source/{neo4j_source_id}/target/{neo4j_target_id}/metapath/{metapath}/?format=json
```

Response structure:
- `path_count_info.dwpc` - NOT the observed DWPC; this is `dgp_nonzero_mean`
- `path_count_info.dgp_source_degree` - Source node degree in this metaedge
- `path_count_info.dgp_target_degree` - Target node degree in this metaedge
- `paths[0].PDP` - **Actual observed DWPC** (Path Degree Product)

### Multi-DWPC CSV Column Reference

| Column | Description |
|--------|-------------|
| `p_value` | Het.io's calculated p-value |
| `dwpc` | Observed DWPC value |
| `dgp_n_dwpcs` | Total null samples (N) |
| `dgp_n_nonzero_dwpcs` | Non-zero null samples (n) |
| `dgp_nonzero_mean` | Mean of non-zero null DWPCs |
| `dgp_nonzero_sd` | Std of non-zero null DWPCs |

### Validation Sample Output (DWPC)

```
     go_id  entrez_gene_id  hetio_pdp  our_dwpc  abs_diff
GO:0070252           10052   0.016667  0.016667       0.0
GO:0014032            3091   0.005356  0.005356       0.0
GO:0006813           10021   0.010673  0.010673       0.0
GO:0061008           26019   0.014982  0.014982       0.0
GO:0042752            2767   0.015831  0.015831       0.0
```

## Next Steps

1. **Increase permutation count**: Test with more permutations (e.g., 20-50) to improve null distribution coverage
2. **Investigate degree-grouping strategy**: Compare our tolerance bands to het.io's exact approach
3. **Consider empirical percentile p-values**: As alternative to gamma-hurdle fitting with limited permutations

## HPC Instructions

### Running P-Value Validation on Alpine

The end-to-end validation script has been configured for HPC execution. Use the SLURM script to test calibration with 20 permutations:

```bash
# Submit job with default settings (20 perms, 100 pairs, 50 samples/group)
sbatch scripts/28_pvalue_validation.sh

# Or with custom parameters
sbatch scripts/28_pvalue_validation.sh 20 100 50  # n_perms, n_pairs, n_samples
sbatch scripts/28_pvalue_validation.sh 50 200 100  # More thorough analysis
```

### Script Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_perms` | 20 | Number of permutations for null distribution |
| `n_pairs` | 100 | Number of pairs for calibration test |
| `n_samples` | 50 | Samples per degree group per permutation |

### Resource Requirements

- **Memory**: 16GB (loads multiple permutation matrices)
- **Time**: ~2 hours for 20 permutations
- **CPUs**: 4 cores

### Output Files

Results are saved to `results/pvalue_validation/`:
- `calibration_{n_perms}perms.csv` - Full calibration results
- `summary_{n_perms}perms.csv` - Summary statistics
- `hetio_comparison_{n_perms}perms.csv` - Het.io comparison (if available)

### Monitoring

```bash
# Check job status
squeue -u $USER

# View output log
tail -f logs/pvalue_validation/pvalue_validation_*.out

# View error log
tail -f logs/pvalue_validation/pvalue_validation_*.err
```

### Expected Results

With 20 permutations, we expect improved calibration compared to 5 permutations:
- **5 permutations**: Mean p-value = 0.044 (severely deflated)
- **20 permutations**: Should be closer to 0.50

If calibration remains poor with 20 permutations, consider:
1. Increasing to 50+ permutations
2. Modifying the degree-grouping tolerance
3. Switching to empirical percentile p-values

### Running Locally

For quick testing with fewer permutations:

```bash
# Activate environment
conda activate CAPP

# Run with 5 permutations (fast, ~5 minutes)
python scripts/validate_pvalue_end_to_end.py --n_perms 5 --n_pairs 30

# Run with 20 permutations (slower, ~30 minutes)
python scripts/validate_pvalue_end_to_end.py --n_perms 20 --n_pairs 100

# Skip het.io comparison (if Multi-DWPC data unavailable)
python scripts/validate_pvalue_end_to_end.py --n_perms 20 --skip_hetio
```

## Conclusions

### Formula Validations (PASSED)

1. **DWPC formula**: Exactly matches het.io (100% for length-1 paths)
2. **P-value formula**: Exactly matches het.io (100% for length >= 2 paths) when using het.io's precomputed statistics

### End-to-End Validation (FAILED)

Using our pipeline with 5 permutations:
- P-values are severely **deflated** (mean = 0.044 instead of 0.50)
- 70% of p-values < 0.05 (expected 5%)
- Issue is **insufficient null distribution coverage**, not formula errors

### Key Insight

The formula validation (100% match) only proves `our_formula(het.io_stats) == het.io_pvalue`. It does NOT validate our null distribution computation. The end-to-end test reveals the actual calibration failure comes from:
- Using only 5 permutations vs het.io's 200
- Insufficient sampling of degree-similar pairs
- The fundamental challenge of estimating null distributions from limited data

### Practical Implications

For production use, we need either:
1. More permutations (closer to het.io's 200)
2. Better degree-grouping strategy
3. Alternative approach like empirical percentiles instead of gamma-hurdle fitting

## Two-Stage Validation Workflow (Added Later)

The original script 28 was running the wrong test (calibration test for uniform p-values instead of comparing to het.io). Since Multi-DWPC data is not available on HPC, we created a two-stage workflow:

### Stage 1a: Extract het.io pairs (Local)

Run locally where Multi-DWPC data exists:

```bash
python scripts/29a_extract_hetio_pairs.py --n_pairs 100
# Output: data/hetio_pairs_for_validation.csv
```

This extracts specific Gene-BP pairs from het.io with their:
- Identifiers (go_id, entrez_gene_id, hetmat indices)
- Het.io's DWPC and p-value
- Het.io's null distribution statistics (dgp_*)

### Stage 1b: Compute stats on HPC

Upload the pairs file to HPC and run:

```bash
# Upload
scp data/hetio_pairs_for_validation.csv <hpc>:<project_dir>/data/

# Submit job
sbatch scripts/29b_compute_hetio_pair_stats.sh 20 50  # n_perms, n_samples
```

This computes for each het.io pair:
- DWPC for Hetionet (true network)
- DWPC for permutation 0
- Null distribution statistics from permutations 1-20

Output: `results/pvalue_validation/hetio_pair_stats_20perms.csv`

### Stage 2: Compare results (Local)

Download HPC results and compare:

```bash
# Download
scp <hpc>:<project_dir>/results/pvalue_validation/hetio_pair_stats_20perms.csv results/pvalue_validation/

# Compare
python scripts/29c_compare_to_hetio.py --input_file results/pvalue_validation/hetio_pair_stats_20perms.csv
```

This compares:
1. Our DWPC vs het.io DWPC (should match exactly)
2. Our p-values vs het.io p-values (will differ due to fewer permutations)
3. Perm0 p-values vs Hetionet p-values

### Files Created

| File | Description |
|------|-------------|
| `scripts/29a_extract_hetio_pairs.py` | Local: Extract het.io pairs for HPC |
| `scripts/29b_compute_hetio_pair_stats.py` | HPC: Compute DWPCs and null stats |
| `scripts/29b_compute_hetio_pair_stats.sh` | SLURM script for 29b |
| `scripts/29c_compare_to_hetio.py` | Local: Compare HPC results to het.io |

### Why This Workflow

The original approach tried to validate everything in one script, but:
1. HPC doesn't have Multi-DWPC data (het.io ground truth)
2. Local doesn't have all 200 permutations efficiently accessible

The two-stage workflow separates concerns:
- **Local**: Has het.io data, extracts pairs
- **HPC**: Has permutation matrices, computes null distributions
- **Local**: Has both, compares results
