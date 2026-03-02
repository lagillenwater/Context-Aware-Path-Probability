# Notebook 11.2 Update: Degree Product Comparison

## Summary

Updated notebook 11.2 to add comprehensive degree product analysis showing how intermediate gene degrees affect compositional predictions.

## Changes Made

### Cell 7: Enhanced `compute_metapath_compositionality()` Function

**Added intermediate gene degree calculations:**
```python
# Compute gene degrees from both edge types
gene_degrees_in = np.array(edge1_aligned.sum(axis=0)).flatten()   # From CbG
gene_degrees_out = np.array(edge2_aligned.sum(axis=1)).flatten()  # From GpPW
```

**Added two pathway degree product metrics:**

1. **Total Connectivity**: Product of (in-degree + out-degree) for each gene
   ```python
   gene_degree_total = gene_in + gene_out
   pathway_degree_product_total *= gene_degree_total
   ```

2. **Joint Connectivity**: Product of (in-degree × out-degree) for each gene
   ```python
   gene_degree_joint = gene_in * gene_out
   pathway_degree_product_joint *= gene_degree_joint
   ```

**New DataFrame columns added:**
- `pathway_degree_product_total`: ∏_genes(in_degree + out_degree)
- `pathway_degree_product_joint`: ∏_genes(in_degree × out_degree)

### Cell 13: Replaced 1×2 Grid with 1×3 Degree Product Comparison

**Three scatter plots created:**

1. **Left Plot: Endpoints Only**
   - Formula: `degree_product_endpoints = C × PW`
   - Colormap: `coolwarm`
   - Shows: Baseline without intermediate genes

2. **Middle Plot: Total Connectivity**
   - Formula: `degree_product_total = C × PW × ∏(in+out)`
   - Colormap: `viridis`
   - Shows: Impact of total gene connectivity (sum of degrees)

3. **Right Plot: Joint Connectivity**
   - Formula: `degree_product_joint = C × PW × ∏(in×out)`
   - Colormap: `plasma`
   - Shows: Impact of hub genes (product of degrees)

**All plots show:**
- X-axis: Compositional Probability (Option A)
- Y-axis: Observed Frequency
- Colors: log10(degree product + 1)
- Reference line: y=x (perfect compositional fit)

## Interpretation Guide

### What Each Metric Captures

**Endpoints Only (C × PW)**:
- Baseline degree product ignoring intermediate genes
- Only considers compound and pathway connectivity

**Total Connectivity (∏(in+out))**:
- Measures overall gene importance
- Gene with in=5, out=10 contributes factor of 15
- Captures total degree regardless of direction

**Joint Connectivity (∏(in×out))**:
- Identifies hub genes (high connectivity in BOTH directions)
- Gene with in=5, out=10 contributes factor of 50
- Emphasizes genes that are hubs for both compounds AND pathways

### Visual Analysis

**Compare color gradients across plots:**
- Do high degree products (red/yellow) cluster differently?
- Which formulation shows clearest separation of observed frequencies?
- Are outliers from y=x line associated with specific degree patterns?

**Key Questions:**
1. Does middle plot show more structure than left → total connectivity matters
2. Does right plot show more structure than middle → hub genes are key
3. Are all three similar → endpoint degrees dominate, genes don't add much

## Expected Insights

The comparison reveals whether:
- **Intermediate gene degrees** affect pathway prediction accuracy
- **Total connectivity** (sum) or **joint connectivity** (product) is more predictive
- **Hub genes** (high in×out) create distinct patterns in compositional deviations

## Files Modified

- **notebooks/11.2_empirical_vs_analytical_compositional.ipynb**
  - Cell 7: Added pathway degree product calculations
  - Cell 13: Replaced 1×2 grid with 1×3 degree product comparison

## Output

**New figure**: `results/option_a_degree_product_comparison.png`
- 1×3 grid of scatter plots
- 24×8 inch figure (high resolution)
- Saved at 300 DPI

## Mathematical Formulation

For metapath **Compound → Gene → Pathway** (CbGpPW):

**Endpoints Only:**
```
degree_product = deg(compound) × deg(pathway)
```

**Total Connectivity:**
```
degree_product = deg(compound) × deg(pathway) × ∏_{g∈shared_genes} [deg_in(g) + deg_out(g)]
```

**Joint Connectivity:**
```
degree_product = deg(compound) × deg(pathway) × ∏_{g∈shared_genes} [deg_in(g) × deg_out(g)]
```

Where:
- `deg_in(g)`: Number of compounds binding to gene g (from CbG edge)
- `deg_out(g)`: Number of pathways gene g participates in (from GpPW edge)
- `shared_genes`: Genes connecting specific compound-pathway pair

## Next Steps

After running the updated notebook:
1. Compare color distributions across the three plots
2. Calculate correlations between degree products and residuals (observed - predicted)
3. Determine if including intermediate gene degrees improves compositional model fit
4. Consider applying same analysis to notebook 11.3 (full 20-permutation version)
