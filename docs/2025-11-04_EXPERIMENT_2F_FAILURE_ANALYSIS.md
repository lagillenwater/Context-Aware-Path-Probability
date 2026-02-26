# Experiment 2F Failure Analysis
Date: 2025-11-04

## Summary

Experiment 2F attempted degree-stratified composition for null model prediction of CbGiGpPW pathway counts. Despite both base models achieving r > 0.95 (CbGiG: r=0.9506, GiGpPW: r=0.9893), the compositional approach failed with r=0.599 and 55x overprediction.

## Approach

Degree-stratified composition aggregates predictions over G2 degree bins:

```python
for deg_G2 in np.unique(gene_degrees[genes_to_PW]):
    n_G2 = np.sum(gene_degrees[genes_to_PW] == deg_G2)
    pred_CbGiG = model_CbGiG.predict([deg_C, deg_G2, ...])
    pred_GiGpPW = model_GiGpPW.predict([deg_G2, deg_PW, ...])
    contrib = n_G2 * pred_CbGiG * pred_GiGpPW
```

## Results

Performance metrics:
- Correlation: r = 0.599 (FAILURE, worse than Exp 2E)
- Mean true: 3.13
- Mean predicted: 172.11
- Overprediction: 55x
- Avg G2 degrees used: 33.8

Comparison to baselines:
- Exp 2E-v1 (actual CbGiG × freq): r = 0.855, 10x underprediction
- Exp 2E-v2 (pred CbGiG × freq): r = 0.749, 11x underprediction
- Exp 2F (degree-stratified): r = 0.599, 55x OVERprediction

## Root Cause Analysis

### 1. Topology vs Degree Mismatch

Actual pathway decomposition shows extreme sparsity:
- Avg G2 degrees contributing: 3.0
- Max G2 degrees: 13
- Hub genes (deg=8611) frequently serve as intermediates

Degree-stratified approach loses this sparsity:
- Avg G2 degrees used: 33.8 (11x more)
- Includes all genes of each degree that connect to PW
- Does not account for which genes also connect to C

### 2. Topology-Degree Contradiction

The approach mixes topology-specific and degree-only information:
- `n_G2` = topology-specific (which genes connect to PW)
- `pred_CbGiG` = degree-based expected count
- `pred_GiGpPW` = degree-based expected count

This double-counts the topology constraint. We filter genes that connect to PW (topology), then multiply by expected counts (degree), assuming all n_G2 genes contribute equally.

### 3. Count Models vs Probability Models

Both models predict EXPECTED counts in permuted graphs:
- `E[CbGiG | deg_C, deg_G2]`: expected count from C to genes of deg_G2
- `E[GiGpPW | deg_G2, deg_PW]`: expected count from G2 to PW

Multiplying expected counts: `n_G2 × E[CbGiG] × E[GiGpPW]`

This assumes independence and linearity that don't hold for sparse pathways.

### 4. Comparison to Exp 2E Approaches

Exp 2E used topology-informed aggregation:
```python
for gene_G2 in genes_connected_to_both_C_and_PW:
    contrib = CbGiG_count[C, G2] * P_edge[deg_G2, deg_PW]
```

Key differences:
- Exp 2E: Only aggregates over genes that actually connect to BOTH C and PW
- Exp 2E: Avg 2.5 intermediates (actual topology)
- Exp 2F: Aggregates over all genes of each degree that connect to PW
- Exp 2F: Avg 33.8 intermediates (11x more)

Exp 2E underpredicted (10x) because edge frequencies are tiny probabilities (0.003-0.02).
Exp 2F overpredicted (55x) because it includes many genes that don't actually connect to C.

## G1 Degree Analysis

From decomposition data, G1 degrees are preserved by G2 degree:
- Row 5: C=348, PW=226, deg_C=9, deg_PW=17, actual=6
  - G2 deg=8611: G1 degrees [4, 11, 2, 7, 7, 4] (mostly low degree)
- Row 8: C=80, PW=586, deg_C=11, deg_PW=105, actual=11
  - 7 different G2 degrees, diverse G1 distributions

This suggests G1 degree matters, but the bigger issue is sparsity and specificity of connections.

## Fundamental Problem with Degree-Stratified Null

For a purely degree-based null model:
- Cannot use topology to filter intermediates (defeats null model purpose)
- Must aggregate over all possible intermediates of each degree
- Loses sparsity and specificity of actual connections
- Results in massive overprediction

For accurate compositional prediction:
- Need to know which specific genes connect C and PW (topology-dependent)
- But this requires enumerating actual pathways, which we're trying to avoid
- Catch-22: composition requires topology, but null model excludes topology

## Conclusion

Degree-stratified composition for null models is fundamentally limited:
1. Pure degree-based approach loses critical sparsity (overpredicts 55x)
2. Topology-informed approach achieves better performance (r=0.855) but requires enumeration
3. For 3-edge pathways (CbGiGpPW), compositional null prediction may not be viable

Alternative approaches to consider:
- Direct 3-edge null models (train on CbGiGpPW counts from permutations)
- Hybrid models with topology features beyond just degree
- Sparsity-aware composition with false positive suppression
- Path-specific models rather than universal composition

## Files Generated

- `test_src/test_degree_stratified_composition_exp2f.py`: Implementation
- `results/hierarchical_prediction/experiment2f_results.csv`: Summary metrics
- `results/hierarchical_prediction/experiment2f_decomposition_analysis.csv`: First 100 pairs decomposed by G2 degree
