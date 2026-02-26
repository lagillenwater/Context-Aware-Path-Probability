# Hierarchical Path Prediction Results - November 4, 2025

## Overview

This document reports results from testing whether long pathway counts can be predicted from shorter pathway counts using linear regression with rich feature sets. This approach differs from the failed edge probability multiplication method (notebook 17, r=0.35) by using actual counts instead of probabilities, including node degree features, and learning optimal weights through regression.

## Experimental Design

### Core Hypothesis
Long pathway counts exhibit compositional structure at the path level when modeled with appropriate features. If true, we can compute only short paths directly (fast) and predict long paths via trained models (instantaneous).

### Success Criteria
- Success: r > 0.95 (prediction is viable for replacing direct counting)
- Partial: 0.8 < r < 0.95 (prediction may be useful for approximation)
- Failure: r < 0.8 (must use direct counting or alternative approaches)

---

## Experiment 1: Length-2 from Length-1 (Sanity Check)

### Objective
Test whether we can predict length-2 pathway counts from length-1 edge counts using node-level aggregated features. This should succeed with very high correlation as it is nearly tautological.

### Metapath
CbGaD (Compound→Gene→Disease)
- Edge 1: CbG (Compound binds Gene) - 11,571 edges
- Edge 2: GaD (Gene associates Disease) - reverse of DaG with 12,623 edges

### Method

**Data:**
- Sample 10,000 node pairs (5,000 train, 5,000 test)
- Target ratio: 50% pairs with non-zero paths, 50% random pairs
- Ground truth: Pathway counts computed via sparse matrix multiplication (CbG @ GaD)

**Features:**
1. Aggregated edge counts per node
   - Total CbG edges from source compound
   - Total GaD edges to target disease
2. Node degrees
   - Compound degree in CbG network
   - Disease degree in DaG network
3. Interaction terms
   - Product of edge counts: Count(CbG) × Count(GaD)
   - Product of degrees: deg_C × deg_D
4. Polynomial terms
   - deg_C²
   - deg_D²

**Model:** Linear regression (ordinary least squares)

### Results

**Performance Metrics:**
- Train: r = 0.306, R² = 0.094, MAE = 0.45
- Test: r = 0.319, R² = 0.102, MAE = 0.44
- Status: **FAILURE** (r < 0.8)

**Top Feature Coefficients:**
- count_CbG: 0.0107
- deg_C: 0.0107
- count_GaD: 0.0063
- deg_D: 0.0063
- Interaction terms: near zero (ineffective)

**Computational Cost:**
- Direct counting: 0.0004s
- Model training: 0.0011s
- Prediction: 0.0003s

### Interpretation

**The experiment FAILED with r = 0.32, which is essentially the same as notebook 17's r = 0.35.**

**Root Cause Analysis:**

The fundamental flaw is that we're trying to predict **pair-specific pathway counts** from **node-level aggregate features**.

To predict Count(Compound_C → Gene → Disease_D), our features are:
- Total edges from Compound C (aggregate across all genes)
- Total edges to Disease D (aggregate across all genes)
- Interaction of these aggregates

**But this doesn't capture which specific intermediate genes connect C to D!**

Two compounds might have the same total degree but connect to completely different sets of genes. Similarly for diseases. The aggregate features lose the critical information about shared intermediate nodes.

**Why This Differs from Notebook 17 But Still Fails:**
- Notebook 17: Used edge probabilities, assumed independence
- Experiment 1: Used edge counts, included degrees
- **Both fail because:** They don't capture the specific connectivity pattern between node pairs

**This is NOT just a technical bug - it's a fundamental conceptual error in the experimental design.**

---

## Experiment 2A: Length-3 from Length-2 Subpath Counts

### Objective
Test whether aggregated length-2 subpath counts contain sufficient information to predict length-3 pathway counts through learned composition.

### Metapath
CbGiGpPW (Compound→Gene→Gene→Pathway)
- Subpath 1: CbGiG (Compound→Gene→Gene) - 177,489 non-zero pairs
- Subpath 2: GiGpPW (Gene→Gene→Pathway) - 3,213,672 non-zero pairs
- Full path: CbGiGpPW - 796,018 non-zero pairs

### Method

**Subpath Computation** (fast with sparse matrices):
```python
CbGiG = CbG @ GiG          # 0.004s
GiGpPW = GiG @ GpPW        # 0.042s
CbGiGpPW = CbGiG @ GpPW    # 0.011s (ground truth)
```

**Features** (13 features per pair):
1. Aggregated subpath statistics:
   - total_CbGiG: Sum of all CbGiG counts from source
   - total_GiGpPW: Sum of all GiGpPW counts to target
   - max_CbGiG, max_GiGpPW: Maximum intermediate counts
   - n_nonzero_CbGiG, n_nonzero_GiGpPW: Number of active intermediates
2. Endpoint degrees: deg_C, deg_PW
3. Interaction terms: total_CbGiG × total_GiGpPW, deg_C × deg_PW
4. Naive composition: sum(CbGiG[i,:] × GiGpPW[:,j]) over intermediates
5. Polynomial terms: deg_C², deg_PW²

**Model**: Linear regression on 5,000 train pairs

### Results

**Performance Metrics:**
- Train: r = 0.601, R² = 0.362, MAE = 0.33
- Test: r = 0.619, R² = 0.382, MAE = 0.33
- Status: **FAILURE** (r < 0.70 threshold)

**Baseline Comparisons:**
- Degrees only: r = 0.404 (like Experiment 1)
- Naive composition: r = 0.340 (like notebook 17)
- Full model (subpath features): r = 0.619

**Top Features by Importance:**
1. max_CbGiG: 0.443 (dominant predictor)
2. max_GiGpPW: 0.271
3. deg_C: 0.011
4. naive_composition: 0.002 (near zero!)
5. deg_PW: 0.001

**Computational Cost:**
- Subpath computation: 0.046s
- Model training: 0.003s
- Total: 0.049s vs 0.011s for direct counting

### Interpretation

**Experiment 2A achieved r = 0.62, which is:**
- 94% better than Experiment 1 (r=0.32)
- 53% better than degree-only baseline (r=0.40)
- 82% better than naive composition (r=0.34)
- **But still fails to meet the r > 0.70 threshold for practical use**

**Key Finding: The naive composition coefficient is near zero (0.002)**

This is critical: the feature that directly sums CbGiG × GiGpPW over intermediates (which should capture the compositional structure) has essentially no predictive power. Instead, the model relies on MAX intermediate counts, not sums or products.

**Why This Still Fails:**

1. **Aggregation loses critical information**: Summing or maximizing over intermediates discards the specific pattern of which intermediates are shared
2. **Maximum != sum**: The model finds that the MAXIMUM intermediate count is more predictive than the SUM, suggesting path counts depend on bottleneck intermediates, not total potential paths
3. **Missing pair-specific structure**: Even with subpath counts, we don't know which specific intermediate nodes connect source to target

**Comparison to Yesterday's Success (r > 0.95):**

Yesterday's models achieved r > 0.95 because they predicted pathway counts from **pair-specific degrees in the same permutation**. The key was predicting for the SAME graph realization.

Today's experiments fail because we're trying to predict pathway counts by aggregating information across different intermediates, losing the specific connectivity pattern.

---

## Experiment 2B: Degree-Stratified Compositional Prediction

### Objective

Test whether compositional multiplication works when stratified by intermediate node degrees, using predictions from trained degree-based models.

### Metapath

CbGiGpPW (Compound→Gene→Gene→Pathway)
- Same as Experiment 2A

### Method

**Three-Stage Approach:**

1. Train subpath models using degree features (like yesterday's r > 0.95 success):
   - Model CbGiG: (deg_C, deg_G) → CbGiG count
   - Model GiGpPW: (deg_G, deg_PW) → GiGpPW count

2. For each (Compound_C, Pathway_PW) pair, vectorized prediction:
   - Iterate over all possible intermediate gene degrees d (1 to 8611)
   - Predict: `predicted_CbGiG_to_deg_d` using model_CbGiG(deg_C, d)
   - Predict: `predicted_GiGpPW_from_deg_d` using model_GiGpPW(d, deg_PW)
   - Compute contribution: `contrib[d] = predicted_CbGiG[d] × predicted_GiGpPW[d]`
   - Sum: `predicted_CbGiGpPW = Σ_d contrib[d]`

3. Compare to actual CbGiGpPW counts

**Computational Efficiency:**
- Vectorized: ~400 predictions per pair (2 × 200 degrees)
- For 5,000 pairs: completed in 2.26 seconds (0.5ms per pair)

### Results

**Performance Metrics:**
- Test: r = -0.090, R² = -20,419,726, MAE = 2052
- Status: **CATASTROPHIC FAILURE**

**Prediction Statistics:**
- Predicted range: [107, 3143], mean = 2052, median = 2121
- Actual range: [0, 1.0]
- All predictions are 2000-3000× too large

**Subpath Model Performance:**
- Model CbGiG training: r = 0.67
- Model GiGpPW training: r = 0.64

**Computational Cost:**
- Subpath model training: 0.005s total
- Prediction time: 2.26s for 5,000 pairs
- Much faster than original Exp 2B (killed after 10+ minutes)

### Interpretation

**Experiment 2B achieved r = -0.09, which is:**
- Worse than Experiment 1 (r=0.32) by 128%
- Worse than Experiment 2A (r=0.62) by 115%
- Worse than naive composition (r=0.34) by 127%
- **Catastrophically wrong with negative correlation**

**Root Cause: Massive Overestimation from Summing All Degrees**

The compositional sum across ALL gene degrees creates systematic overestimation:

```python
predicted = sum(pred_CbGiG[d] × pred_GiGpPW[d] for d in 1..8611)
```

**Why this fails:**

1. **Models predict non-zero for all degrees**: The trained models give small but non-zero predictions for all (deg_C, d) and (d, deg_PW) combinations

2. **Sum across thousands of terms**: Summing 8611 small products creates huge totals
   - Even if each product is just 0.5, summing 8611 gives ~4300
   - Actual pathway counts are typically 0-1

3. **Most degrees aren't relevant**: For a specific (C, PW) pair, only a tiny fraction of intermediate gene degrees actually contribute paths
   - The compositional formula treats all degrees as if they contribute equally
   - No way to know which degrees are actually relevant without looking at connectivity

**Comparison to Experiment 2A:**
- Exp 2A used ACTUAL subpath counts (knew which intermediates existed)
- Exp 2B uses PREDICTED counts for ALL possible degrees (no knowledge of actual connectivity)
- Exp 2A failed (r=0.62) because aggregation lost information
- Exp 2B fails worse (r=-0.09) because it aggregates over irrelevant degrees too

**The Fundamental Problem:**

Degree-stratified composition requires knowing:
1. Which intermediate degrees are ACTUALLY present in the path
2. How many intermediates of each degree exist
3. The specific connectivity pattern

Simply predicting counts for all possible degrees and summing fails because:
- We don't know which degrees are relevant
- We overcount by including irrelevant degrees
- The predictions compound errors across thousands of terms

**What This Tells Us:**

Even when stratified by degree (which should capture the key structural variable), compositional multiplication FAILS for predicting pathway counts. This suggests:
- Path counts don't compose multiplicatively, even conditionally
- Need pair-specific information, not just degree statistics
- Length-3+ paths may have fundamentally different structure than length-2

---

## Experiment 2C: Predicted Counts × Edge Probabilities

### Objective

Fix Experiment 2B's overestimation by using edge PROBABILITIES instead of predicted COUNTS for the second term. Test both analytical (configuration model) and empirical (actual graph frequencies) probabilities.

### Metapath

CbGiGpPW (Compound→Gene→Gene→Pathway)
- Same as Experiments 2A and 2B

### Method

**Modified compositional formula:**

Instead of 2B's approach:
```python
contrib[d] = predicted_CbGiG[d] × predicted_GiGpPW[d]  # count × count
```

Use:
```python
contrib[d] = predicted_CbGiG[d] × P(gene_deg_d → PW_deg_pw)  # count × probability
```

**Two Variants:**

**Variant 2C-v1 (Analytical):**
- Use configuration model formula for edge probabilities
- `P(gene_d → PW) = (deg_gene * deg_PW) / n_edges`
- Bounded [0, 1] prevents massive overestimation from 2B

**Variant 2C-v2 (Empirical):**
- Calculate empirical frequencies from actual GpPW graph
- For each (degree_gene, degree_PW) pair:
  - Count genes with degree_gene that connect to pathways with degree_PW
  - Divide by total possible connections
- Uses actual graph structure

### Results

**Variant 1 (Analytical):**
- Test: r = 0.202, R² = -9.51, MAE = 0.67
- Predicted range: [0.002, 25.1], mean = 0.74
- Status: **FAILURE**

**Variant 2 (Empirical):**
- Test: r = 0.288, R² = -314.12, MAE = 4.44
- Predicted range: [0.04, 79.7], mean = 5.02
- Status: **FAILURE** (but better than v1)

**Actual values:**
- Range: [0, 1.0], mean = 0.64

**Computational Cost:**
- Empirical prob computation: Pre-computed for 24,990 degree pairs
- Prediction time (analytical): 10.08s for 5,000 pairs
- Prediction time (empirical): 5.61s for 5,000 pairs

### Interpretation

**Experiment 2C achieved r = 0.29 (empirical, best variant), which is:**
- Better than Experiment 2B (r=-0.09) by fixing negative correlation
- Still worse than Experiment 1 (r=0.32) by 10%
- Much worse than Experiment 2A (r=0.62) by 53%
- **Still fails to achieve useful predictions (r < 0.70)**

**Why probabilities helped but still failed:**

**Improvement over 2B:**
1. Probabilities bounded [0,1] prevent catastrophic overestimation
2. Analytical predictions went from ~2000 to ~0.7 (more reasonable scale)
3. Achieved positive correlation instead of negative

**But still fails because:**
1. **Still overestimates**: Empirical mean = 5.0 vs actual = 0.64
2. **Sums over irrelevant degrees**: Including all 8611 gene degrees when most don't connect specific (C, PW) pairs
3. **Probabilities don't capture pair-specific structure**: P(gene_d → PW) is an average, doesn't know which specific genes connect

**Why empirical > analytical:**
- Empirical uses actual graph frequencies, captures real connectivity patterns better
- Analytical uses theoretical approximation (configuration model) which oversimplifies
- But both still aggregate over too many irrelevant intermediates

**The Core Problem Remains:**

Even with probabilities, degree-stratified composition fails because:
- We don't know which specific intermediate genes are relevant for a given (C, PW) pair
- Summing contributions from all possible degrees includes noise from irrelevant intermediates
- The compositional assumption (multiply and sum) doesn't capture the true structure of path counts

**What Variant Comparison Tells Us:**

Plot 3 shows analytical vs empirical predictions correlate well (systematic relationship), but both systematically overestimate. This means:
- The two methods agree on relative predictions (rank ordering)
- But both miscalibrate absolute magnitudes
- Using actual graph frequencies (empirical) helps but isn't sufficient

---

## Experiment 2D: Base Model Validation (Critical Correction)

### Objective

Before proceeding with hierarchical composition experiments, validate that individual length-2 metapaths (CbGiG and GiGpPW) can be predicted with r > 0.95 using yesterday's proven methodology. This serves as a prerequisite for any hierarchical approach.

### Background

User correctly identified that initial Experiment 2D attempts achieved only r = 0.49-0.69, failing to replicate yesterday's r > 0.95 success. Investigation revealed three critical errors in the initial implementation:

1. **Boolean matrix multiplication**: Edge matrices stored as dtype=bool, causing pathway counts to be binary (0/1) instead of actual counts
2. **Wrong permutation file paths**: Missing `/edges/` subdirectory in permutation file paths
3. **Incorrect validation methodology**: Evaluated on individual permutations instead of mean of permutations 6-20

### Corrected Methodology

Exact replication of yesterday's approach (from `test_src/run_minimum_perms_comparison.py`):

**Data:**
- Sample 10,000 node pairs (50% non-zero paths, 50% random)
- Training target: Pathway counts from permutation 0
- Validation target: MEAN pathway counts from permutations 6-20
- Same pairs used for both training and validation targets
- 80/20 train/test split on the pairs

**Features:**
- Exactly 5 degree features: d_u, d_v, d_u×d_v, d_u², d_v²
- Extracted from permutation 0
- No log transforms (initial attempt incorrectly added 3 extra features)

**Model:**
- Linear regression (OLS)
- Trained to predict validation target (mean of perms 6-20)
- Evaluated on held-out pairs

**Key fixes:**
1. Cast matrices to int32: `sp.load_npz(file).astype(np.int32)`
2. Correct permutation paths: `data/permutations/XXX.hetmat/edges/EdgeType.sparse.npz`
3. Validate on MEAN of perms 6-20, not individual permutations

### Results: CbGiG (Compound → Gene → Gene)

**Data characteristics:**
- CbG: 11,571 edges
- GiG: 294,328 edges
- CbGiG: 212,934 non-zero pathway counts
- Pathway count range: [0, 39]
- Perm 0 mean: 0.545, Val mean (6-20): 0.082

**Performance:**
- Target correlation (perm 0 vs mean 6-20): r = 0.7057
- Model train r (perm 0): r = 0.6675
- Model test r vs perm 0: r = 0.9006
- **Model test r vs mean 6-20: r = 0.9506**
- **Status: VALIDATED (r > 0.95)**

**Files:**
- Script: `test_src/validate_cbgig_model_exp2d_corrected.py`
- Results: `results/hierarchical_prediction/experiment2d_corrected_summary.csv`
- Plots: `results/hierarchical_prediction/experiment2d_corrected_plots.png`

### Results: GiGpPW (Gene → Gene → Pathway)

**Data characteristics:**
- GiG: 294,328 edges
- GpPW: 84,372 edges
- GiGpPW: 3,983,866 non-zero pathway counts
- Pathway count range: [0, 60]
- Perm 0 mean: 0.878, Val mean (6-20): 0.516

**Performance:**
- Target correlation (perm 0 vs mean 6-20): r = 0.8839
- Model train r (perm 0): r = 0.8722
- Model test r vs perm 0: r = 0.9240
- **Model test r vs mean 6-20: r = 0.9861**
- **Status: VALIDATED (r > 0.95)**

**Files:**
- Script: `test_src/validate_gigppw_model_exp2d.py`
- Results: `results/hierarchical_prediction/experiment2d_gigppw_summary.csv`
- Plots: `results/hierarchical_prediction/experiment2d_gigppw_plots.png`

### Interpretation

**Both base models successfully validated:**
- CbGiG: r = 0.9506 (exceeds threshold)
- GiGpPW: r = 0.9861 (well above threshold)

**Key insights:**

1. **CbGiG is learnable**: Despite Gene→Gene interactions, degree features capture systematic patterns that generalize across permutations

2. **GiGpPW performs better**: Higher r (0.9861 vs 0.9506) likely due to:
   - More edges (GiG: 294K, GpPW: 84K vs CbG: 11K, GiG: 294K)
   - Higher pathway density (mean counts: 0.88 vs 0.55)
   - Stronger target correlation (0.88 vs 0.71)

3. **Methodology matters critically**:
   - Boolean matrices → r = 0.49 (FAILURE)
   - Corrected with int32, proper validation → r = 0.95+ (SUCCESS)
   - Small implementation errors cause catastrophic performance degradation

4. **Models exceed target correlation**:
   - CbGiG: target r = 0.71, model r = 0.95 (35% improvement)
   - GiGpPW: target r = 0.88, model r = 0.99 (11% improvement)
   - Degree features capture systematic structure better than single permutations

**Prerequisites now satisfied:**
Both CbGiG and GiGpPW models achieve r > 0.95, confirming that:
1. Length-2 metapaths with Gene→Gene edges can be predicted accurately
2. The methodology from yesterday generalizes to these specific metapaths
3. Base models are validated for potential hierarchical composition

**However**, this validation does NOT guarantee that hierarchical composition will work. Experiments 1, 2A, 2B, and 2C all failed (r < 0.70) despite using predicted or actual subpath counts. The validation only confirms that the building blocks can be predicted; whether they can be combined hierarchically remains unresolved.

### Comparison to Failed Experiments

**Experiment 2D vs Experiment 2B:**
- 2D: Predict CbGiG directly from endpoint degrees → r = 0.95 (SUCCESS)
- 2B: Compose CbGiGpPW from predicted CbGiG + predicted GiGpPW → r = -0.09 (FAILURE)
- **Conclusion**: Individual metapaths are predictable, but composition doesn't work

This stark contrast highlights that the failure of hierarchical composition is NOT due to poor base model quality. The base models work excellently (r > 0.95). The failure lies in the composition step itself.

---

## Experiment 2E: Intelligent Degree-Stratified Composition

### Objective

Test whether intelligent aggregation over only relevant intermediate nodes enables hierarchical composition. This addresses the key failure mode of Experiment 2B: summing over all 8,611 possible gene degrees when most are irrelevant.

### Core Innovation

**Only aggregate over genes that actually connect to the target pathway**, using:
1. Actual OR predicted CbGiG counts (test both)
2. Empirical edge frequencies for GpPW
3. Sparse composition: typically 2-50 genes instead of 8,611

### Methodology

**For each (Compound_C, Pathway_PW) pair:**

```python
# Get genes that connect to target pathway
genes_connected_to_PW = np.nonzero(GpPW_matrix[:, PW])[0]

# Aggregate only over relevant genes
predicted_CbGiGpPW = 0
for gene_G2 in genes_connected_to_PW:
    # Option 1: Use actual CbGiG count
    CbGiG_count = CbGiG_matrix[C, G2]

    # Option 2: Use predicted CbGiG count
    features = [deg_C, deg_G2, deg_C*deg_G2, deg_C^2, deg_G2^2]
    CbGiG_count = model_CbGiG.predict([features])

    # Empirical edge frequency
    P_edge_GpPW = empirical_freq[(deg_G2, deg_PW)]

    # Compose
    predicted_CbGiGpPW += CbGiG_count * P_edge_GpPW
```

**Key differences from Experiment 2B:**
- 2B: Sum over ALL degrees (8,611 terms)
- 2E: Sum over CONNECTED genes only (2-50 terms)
- 2B: Predict both CbGiG and GiGpPW counts
- 2E: Use actual or predicted CbGiG, predict only GpPW edge probability

### Results: Variant 1 (Actual CbGiG Counts)

**Performance:**
- Correlation: r = 0.8547 (PROMISING)
- MAE: 2.92
- Mean true: 3.30, Mean predicted: 0.38
- Prediction ratio: 0.12 (systematic underprediction)
- Average intermediates: 2.5 genes per pair

**Status: PROMISING but systematic underprediction**

**Why it works better than 2B:**
1. Sparse aggregation (2.5 vs 8,611 genes) eliminates irrelevant noise
2. Uses actual CbGiG counts (no prediction error in first step)
3. Only aggregates over genes that provably connect to target

**Why it still underpredicts:**
- Empirical edge frequencies are conservative (P(edge exists))
- Need expected count contribution, not just P(edge)
- Multiplying counts by small probabilities scales down too much

### Results: Variant 2 (Predicted CbGiG Counts)

**Performance:**
- Correlation: r = 0.7491 (PARTIAL)
- MAE: 2.85
- Mean true: 3.13, Mean predicted: 0.29
- Prediction ratio: 0.093 (worse underprediction)
- Average intermediates: 47.8 genes per pair

**Status: PARTIAL - worse than using actual counts**

**Why predicted CbGiG hurts performance:**

1. **False positives**: CbGiG model (r=0.95) predicts small positive values for many gene pairs where actual count is 0
2. **Diffuse predictions**: Aggregates over 47.8 genes instead of 2.5 actual connections
3. **Noise accumulation**: Each false positive adds a small contribution, but these don't align with true pathway structure
4. **Scale mismatch**: Predicted CbGiG values are calibrated for mean of perms 6-20, not individual pathways

**Sample breakdown** (first pair):
```
True CbGiGpPW: 3
Predicted CbGiGpPW: 0.019

Breakdown:
  - 21 intermediate genes considered (model predicts positive CbGiG for all)
  - Sum of predicted CbGiG: 1.96
  - Average GpPW frequency: 0.0033
  - Product: 1.96 × 0.0033 = 0.019 (100x too small)
```

The model diffuses the prediction across 21 genes when only 2-3 actually matter, and each contribution is tiny due to low edge frequencies.

### Performance by Prediction Magnitude

**Variant 1 (Actual):**
Shows better performance at higher magnitudes, but still underestimates.

**Variant 2 (Predicted):**
```
[0, 0.1):   n=3667, r=0.474, mean_true=1.24,  mean_pred=0.02
[0.1, 1):   n=1029, r=0.265, mean_true=5.36,  mean_pred=0.34
[1, 10):    n=285,  r=0.545, mean_true=15.88, mean_pred=2.59
[10, 100):  n=19,   r=0.649, mean_true=55.26, mean_pred=16.23
```

Correlation improves at higher magnitudes, but underprediction ratio stays around 10x across all ranges.

### Interpretation

**Major improvement over blind aggregation:**
- Exp 2B (all degrees): r = -0.09
- Exp 2E (connected only, actual): r = 0.855
- Exp 2E (connected only, predicted): r = 0.749

Intelligent aggregation transforms catastrophic failure into promising results by reducing aggregation from 8,611 terms to 2.5 relevant terms.

**But still falls short of r > 0.95 because:**

1. **Empirical frequencies underestimate contributions**: P(edge) from permutations is conservative
2. **Prediction error compounds**: Using predicted CbGiG makes it worse (r drops from 0.855 to 0.749)
3. **Sparsity mismatch**: Model predicts for ~48 genes when ~3 actually connect

**Key insight from comparison:**

Using actual CbGiG counts (r=0.855) works much better than predicted (r=0.749). This suggests that even though the CbGiG model achieves r=0.95 in isolation, the prediction errors accumulate destructively when composed hierarchically.

**Why hierarchical composition inherently struggles:**

Even with intelligent aggregation and validated base models (r > 0.95), composition fails to reach r > 0.95 because:
- Base model errors: 5% error in CbGiG becomes larger when aggregated
- Calibration mismatch: Models trained on mean of permutations, used for individual pairs
- False positives: Model predicts weak connections that don't exist, diluting signal
- Multiplication compounds errors: Errors in CbGiG and GpPW multiply

### Files Generated

**Scripts:**
- `test_src/test_hierarchical_intelligent_exp2e.py` - Original with actual CbGiG counts
- `test_src/test_hierarchical_intelligent_exp2e_predicted.py` - Updated with predicted CbGiG

**Results:**
- `results/hierarchical_prediction/experiment2e_results.csv`
- `results/hierarchical_prediction/experiment2e_plots.png`
- `results/hierarchical_prediction/experiment2e_predicted_results.csv`
- `results/hierarchical_prediction/experiment2e_predicted_breakdown.csv`
- `results/hierarchical_prediction/experiment2e_predicted_plots.png`

---

## Experiment 2G: Focused Composition Model with Structural Features

### Objective

Improve Exp 2E (r=0.855) by learning to properly weight the composition term based on endpoint degrees and intermediate sparsity. Test whether a minimal feature set can push correlation above r > 0.95 threshold.

### Motivation

Exp 2E achieved r=0.855 with precise topology filtering (2.5 intermediates per pair) but:
- Systematically underpredicted 10x (mean pred: 0.38, mean true: 3.30)
- Simple linear aggregation `sum(CbGiG × P_edge)` doesn't account for structural variation
- 27% of variance unexplained (R² = 0.73)

Hypothesis: Adding endpoint degrees and sparsity allows the model to learn proper weighting.

### Methodology

**Minimal feature set (8 features):**

```python
features = [
    deg_C,                              # Source degree
    deg_PW,                             # Target degree
    deg_C × deg_PW,                     # Degree product
    deg_C²,                             # Source degree squared
    deg_PW²,                            # Target degree squared
    n_intermediates,                    # Sparsity (genes connecting to both)
    composition_sum,                    # Exp 2E formula: sum(CbGiG × P_edge)
    n_intermediates × composition_sum,  # Interaction
]

model = LinearRegression()
model.fit(features, true_CbGiGpPW)
```

**Key differences from Exp 2E:**
- Exp 2E: Just uses `composition_sum` directly (r=0.855)
- Exp 2G: Learns to weight `composition_sum` based on degrees + sparsity (r=0.969)

### Results

**Performance:**
- Train r: 0.9690
- Test r: 0.9693
- **Status: SUCCESS (r > 0.95)**

**Improvement over baselines:**
- Exp 2E baseline (composition only): r = 0.873
- Exp 2G (with features): r = 0.969
- **Improvement: +0.096**

**Scale metrics:**
- Mean true: 3.23
- Mean pred: 3.28
- Ratio: 1.016 (nearly perfect)
- MAE: 0.78

**Feature importance (by coefficient magnitude):**
1. `composition_sum`: 1.35 (strongest predictor)
2. `n_intermediates`: 1.08 (sparsity is critical)
3. `deg_C`: 0.04 (weak positive)
4. `deg_PW`: -0.003 (weak negative)
5. Other degree terms: negligible (<0.001)

### Interpretation

**Why Exp 2G succeeds where Exp 2E fell short:**

1. **Composition term is correct but needs calibration**: The Exp 2E formula `sum(CbGiG × P_edge)` captures the core signal (coefficient 1.35), but raw application gives wrong scale

2. **Sparsity matters critically**: Coefficient of 1.08 for `n_intermediates` shows that pairs with more intermediates contribute MORE than simple linear aggregation would suggest
   - Sparse pathways (2-3 intermediates): Lower contributions
   - Dense pathways (5-10 intermediates): Higher contributions
   - Simple sum doesn't account for this non-linearity

3. **Endpoint degrees provide weak additional signal**:
   - `deg_C` has small positive effect (0.04)
   - `deg_PW` has negligible negative effect (-0.003)
   - Most degree information already captured by composition term

4. **Nearly perfect scale**: Ratio of 1.016 shows model learns proper calibration to fix Exp 2E's 10x underprediction

**Why this works better than Exp 2F:**
- Exp 2F: Aggregates by degree, loses individual gene information → r=0.60, 55x overprediction
- Exp 2G: Uses actual intermediate genes (like Exp 2E) but learns proper weighting → r=0.97, 1.0x scale

**Key scientific insight:**

Hierarchical composition CAN achieve r > 0.95 when:
1. Use precise topology filtering (which genes actually connect - Exp 2E's contribution)
2. Learn to weight the composition term based on structural features (Exp 2G's contribution)
3. Account for sparsity non-linearity (pairs with more intermediates need upweighting)

The failure of earlier experiments (2B, 2C, Exp 2E raw) was not that composition is fundamentally impossible, but that simple linear aggregation doesn't properly account for structural variation.

### Implications

**For pathway prediction:**
- Length-3 pathways CAN be predicted from length-2 subpaths with r > 0.95
- Requires: (1) actual topology to identify intermediates, (2) learned weighting based on structure
- This approach is VIABLE for replacing direct enumeration when topology is available

**For null models:**
- This success requires knowing which specific genes connect (topology-dependent)
- For null model use case (predicting without topology), Exp 2F showed degree-only aggregation fails (r=0.60)
- May need direct 3-edge null models rather than composition

**Computational implications:**
- Must enumerate intermediates to build features (not faster than direct counting)
- Useful for understanding pathway structure, not computational speedup
- May enable transfer learning or pathway-specific models

### Files Generated

**Scripts:**
- `test_src/test_focused_composition_exp2g.py` - Focused composition with structural features

**Results:**
- `results/hierarchical_prediction/experiment2g_results.csv` - Performance metrics
- `results/hierarchical_prediction/experiment2g_feature_importance.csv` - Feature coefficients

**Visualizations:**
- `results/hierarchical_prediction/experiment2g_plots.png` - Comparison to Exp 2E baseline

---

## Experiment 2F: Degree-Stratified Composition for Null Models

### Objective

Implement a purely degree-based composition approach suitable for null model use case. Unlike Exp 2E which used topology to filter genes, test whether aggregating over G2 degree bins enables compositional prediction without topology information.

### Motivation

For null model anomaly detection, we want to predict pathway counts from 1-5 permutations without enumerating all pathways. This requires predicting E[count | degrees] without using topology-specific information (which genes connect C to PW).

### Methodology

**Degree-stratified aggregation:**

```python
# For each (Compound_C, Pathway_PW) pair:
predicted_CbGiGpPW = 0

# Find all genes that connect to PW (topology-specific filter)
genes_to_PW = np.nonzero(GpPW_matrix[:, PW])[0]

# Aggregate by G2 degree
for deg_G2 in np.unique(gene_degrees[genes_to_PW]):
    n_G2 = np.sum(gene_degrees[genes_to_PW] == deg_G2)

    # Predict counts from both models
    pred_CbGiG = model_CbGiG.predict([deg_C, deg_G2, ...])
    pred_GiGpPW = model_GiGpPW.predict([deg_G2, deg_PW, ...])

    # Aggregate: n_genes × count × count
    contrib = n_G2 * pred_CbGiG * pred_GiGpPW
    predicted_CbGiGpPW += contrib
```

**Key differences from Exp 2E:**
- Uses GiGpPW count model instead of edge frequencies (probabilities)
- Aggregates by G2 degree instead of individual genes
- Uses n_genes × count × count instead of count × probability

**Both base models validated:**
- CbGiG: r = 0.9506 (SUCCESS)
- GiGpPW: r = 0.9893 (SUCCESS)

### Results

**Performance:**
- Correlation: r = 0.5992 (FAILURE)
- MAE: 169.26
- Mean true: 3.13, Mean predicted: 172.11
- Overprediction: 55x (systematic OVER-prediction, not under)
- Avg G2 degrees used: 33.8

**Actual pathway decomposition (first 100 pairs):**
- Avg G2 degrees contributing: 3.0
- Max G2 degrees: 13
- Pathways are extremely sparse

**Comparison to baselines:**
- Exp 2E-v1 (actual CbGiG × freq): r = 0.855, 10x underprediction
- Exp 2E-v2 (pred CbGiG × freq): r = 0.749, 11x underprediction
- Exp 2F (degree-stratified counts): r = 0.599, 55x OVERprediction

**Status: FAILURE (worse than Exp 2E)**

### Root Cause Analysis

**1. Topology-Degree Contradiction**

The approach mixes topology-specific and degree-only information:
- `genes_to_PW`: Topology-specific (which genes connect to PW)
- `n_G2`: Count of genes with this degree (topology-specific)
- `pred_CbGiG`, `pred_GiGpPW`: Degree-based expected counts

This assumes all n_G2 genes of a given degree contribute equally, but only a tiny fraction actually connect to C.

**2. Sparsity Loss**

Actual pathways decompose to 3.0 G2 degrees on average, but the approach aggregates over 33.8 degrees (11x more). By including all genes of each degree that connect to PW, we vastly overestimate because most don't also connect to C.

**3. Expected Counts Don't Compose**

Both models predict expected counts in permuted graphs:
- `E[CbGiG | deg_C, deg_G2]`
- `E[GiGpPW | deg_G2, deg_PW]`

Multiplying: `n_G2 × E[CbGiG] × E[GiGpPW]` assumes independence and linearity that don't hold for sparse pathways.

**4. Fundamental Null Model Limitation**

For a purely degree-based null model:
- Cannot use topology to filter intermediates (defeats null model purpose)
- Must aggregate over all possible intermediates of each degree
- Loses sparsity and specificity of actual connections
- Results in massive overprediction

For accurate composition:
- Need to know which specific genes connect C and PW (topology-dependent)
- But this requires enumerating actual pathways, which we're trying to avoid
- Catch-22: composition requires topology, but null model excludes topology

### Interpretation

**Why Exp 2F failed worse than Exp 2E:**

Exp 2E achieved r=0.855 by aggregating over genes that actually connect to BOTH C and PW (2.5 genes on average). It underpredicted 10x because edge frequencies are small probabilities (0.003-0.02).

Exp 2F achieved r=0.599 by aggregating over all genes of each degree that connect to PW (33.8 degrees on average), regardless of whether they connect to C. It overpredicted 55x because it includes many genes that don't actually participate in pathways.

**Why degree-stratified null models fail for composition:**

1. Pure degree-based approach loses critical sparsity
2. Topology-informed approach achieves better performance but requires enumeration
3. For 3-edge pathways, compositional null prediction may not be viable

**Key insight:**

Even with both base models achieving r > 0.95, degree-stratified composition fails because pathway structure is highly sparse and specific. Only a tiny fraction of genes with a given degree participate in pathways between specific compound-pathway pairs.

### Files Generated

**Scripts:**
- `test_src/test_degree_stratified_composition_exp2f.py` - Degree-stratified composition implementation

**Results:**
- `results/hierarchical_prediction/experiment2f_results.csv` - Performance metrics
- `results/hierarchical_prediction/experiment2f_decomposition_analysis.csv` - First 100 pairs decomposed by G2 degree

**Analysis:**
- `docs/2025-11-04_EXPERIMENT_2F_FAILURE_ANALYSIS.md` - Detailed root cause analysis

---

## Summary and Conclusions

### Experimental Results Overview

| Experiment | Approach | Test r | Status |
|------------|----------|--------|--------|
| Exp 1 | Aggregate degrees → CbGaD | 0.32 | FAILURE |
| Exp 2A | Actual subpath counts → CbGiGpPW | 0.62 | FAILURE |
| Exp 2B | Degree-stratified (count × count) → CbGiGpPW | -0.09 | CATASTROPHIC FAILURE |
| Exp 2C-v1 | Degree-stratified (count × analytical prob) → CbGiGpPW | 0.20 | FAILURE |
| Exp 2C-v2 | Degree-stratified (count × empirical prob) → CbGiGpPW | 0.29 | FAILURE |
| **Exp 2D-CbGiG** | **Direct prediction (5 degree features)** | **0.95** | **SUCCESS** |
| **Exp 2D-GiGpPW** | **Direct prediction (5 degree features)** | **0.99** | **SUCCESS** |
| Exp 2E-v1 | Intelligent aggregation (actual CbGiG × GpPW freq) | 0.85 | PROMISING |
| Exp 2E-v2 | Intelligent aggregation (predicted CbGiG × GpPW freq) | 0.75 | PARTIAL |
| Exp 2F | Degree-stratified composition (count × count by G2 degree) | 0.60 | FAILURE |
| **Exp 2G** | **Focused composition (Exp 2E + degrees + sparsity)** | **0.97** | **SUCCESS** |

### Key Findings

**1. Hierarchical composition CAN work with proper structural features**
- Direct prediction of CbGiG from endpoint degrees: r = 0.95 (SUCCESS)
- Direct prediction of GiGpPW from endpoint degrees: r = 0.99 (SUCCESS)
- Naive hierarchical composition (all degrees): r = -0.09 (CATASTROPHIC FAILURE)
- Intelligent hierarchical composition (connected genes only): r = 0.85 (PROMISING but insufficient)
- Degree-stratified composition (null model approach): r = 0.60 (FAILURE)
- **Focused composition (Exp 2E + degrees + sparsity): r = 0.97 (SUCCESS)**
- **Conclusion**: Composition achieves r > 0.95 when properly weighted by structural features (sparsity + degrees)

**2. Intelligent aggregation dramatically improves composition**
- Exp 2B (all 8,611 degrees): r = -0.09
- Exp 2E (2.5 relevant genes): r = 0.85
- Reducing aggregation by 3,400x transforms catastrophic failure into promising results
- But still falls short of r > 0.95 threshold due to systematic underprediction

**3. Using predicted base models compounds errors**
- Exp 2E with actual CbGiG counts: r = 0.855 (9% underprediction)
- Exp 2E with predicted CbGiG counts: r = 0.749 (91% underprediction)
- Even though CbGiG model achieves r = 0.95 in isolation, errors accumulate destructively when composed
- Model false positives diffuse predictions across 48 genes instead of 3 actual connections

**4. Aggregation fundamentally loses pair-specific structure**
- Experiment 1: Aggregate degrees lose pair-specific connectivity
- Experiment 2A: Aggregating subpath counts loses intermediate alignment
- Experiment 2B: Summing over all degrees includes irrelevant intermediates
- Experiment 2E: Even with intelligent aggregation, empirical frequencies underestimate contributions

**5. Composition requires more than validated base models**
- Base models achieve r > 0.95 when predicting directly
- But r drops to 0.75 when composed hierarchically
- Calibration mismatch: Models trained on mean of permutations, used for individual pathways
- False positives: Model predicts weak connections that don't exist
- Error multiplication: Small errors in each step compound multiplicatively

**6. Degree-stratified null models fail for composition**
- Exp 2F tested purely degree-based composition for null modeling
- Results: r = 0.60, 55x overprediction (worse than Exp 2E)
- Root cause: Cannot filter by topology without defeating null model purpose
- Aggregating over all genes of each degree loses sparsity (3.0 → 33.8 intermediates)
- Conclusion: For 3-edge pathways, compositional null prediction may not be viable

**7. Sparsity and structural weighting are critical for composition**
- Exp 2G shows n_intermediates (sparsity) is the second most important feature (coef=1.08)
- Simple linear aggregation underpredicts because it doesn't account for non-linear effects
- Pairs with more intermediates contribute MORE than linear sum suggests
- Learned weighting based on degrees + sparsity achieves r=0.97 vs r=0.87 for raw composition

**8. Implementation details matter critically**
- Boolean matrices → r = 0.49 (FAILURE)
- Corrected with int32, proper validation → r = 0.95+ (SUCCESS)
- Small errors cause catastrophic performance degradation
- Must validate against known successful methodologies

### Scientific Implications

**Why Some Compositional Approaches Fail:**

1. **Information loss through aggregation**: Both actual and predicted counts lose the specific pattern of which intermediates connect endpoints (Exp 2A, 2B)

2. **Multiplication compounds errors**: When composing predictions, errors multiply rather than average out (Exp 2E-v2 with predicted CbGiG)

3. **Irrelevant intermediates dilute signal**: Including all possible degrees (Exp 2B) or aggregating over unknown intermediates (Exp 2A) introduces noise

4. **Simple linear aggregation underestimates**: Raw sum doesn't account for sparsity effects (Exp 2E: r=0.87)

**Why Exp 2G Succeeds:**

1. **Precise topology filtering**: Only includes genes that actually connect to both endpoints (from Exp 2E)

2. **Learned structural weighting**: Model learns to upweight pathways with more intermediates (sparsity non-linearity)

3. **Proper calibration**: Fixes 10x underprediction by learning endpoint degree effects

4. **Composition term captures core signal**: Exp 2E formula has coefficient 1.35, showing it's fundamentally correct but needs calibration

**Implications for Computational Strategy:**

**Exp 2G shows hierarchical composition CAN achieve r > 0.95**, but with important caveats:

1. **Requires actual topology**: Must identify which specific genes connect C and PW (can't use degree-only predictions)

2. **Not computationally faster**: Must enumerate intermediates to build features (same cost as direct counting)

3. **Useful for understanding, not speedup**: Reveals that sparsity and structural features matter, but doesn't reduce computational cost

**For practical use:**
- Direct sparse matrix multiplication remains necessary (efficient and accurate)
- Exp 2G useful for scientific understanding of pathway structure
- May enable transfer learning or pathway-specific models in future work

**For null models:**
- Exp 2G requires topology (not applicable to null model use case)
- Direct 3-edge null models likely needed (predict CbGiGpPW from degrees without composition)

### Comparison to Notebook 17

Notebook 17 tested compositional multiplication of EDGE PROBABILITIES and achieved r = 0.35. Today's experiments tested a progression of approaches:
- Exp 1: Aggregate node degrees → r = 0.32 (similar failure)
- Exp 2A: Actual pathway counts → r = 0.62 (better but still fails)
- Exp 2B: Predicted degree-stratified counts → r = -0.09 (worse)
- Exp 2E: Intelligent aggregation with topology → r = 0.85 (promising)
- Exp 2G: Learned weighting with structural features → r = 0.97 (SUCCESS)

**Revised Conclusion:** Compositional prediction IS possible (r > 0.95) when combining:
1. Precise topology filtering (which genes actually connect)
2. Learned weighting based on structural features (sparsity + degrees)
3. Proper calibration of the composition term

The key insight: composition requires both correct intermediates (topology) and correct weighting (learned from structural features).

### Files Generated

**Scripts:**
- `test_src/test_hierarchical_prediction_exp1.py` - Aggregate degree features
- `test_src/test_hierarchical_subpath_exp2a.py` - Actual subpath counts
- `test_src/investigate_exp2a_features.py` - Feature value analysis for 2A
- `test_src/test_hierarchical_degrees_exp2b.py` - Original 2B (killed, too slow)
- `test_src/test_hierarchical_degrees_exp2b_optimized.py` - Vectorized count × count
- `test_src/test_hierarchical_degrees_exp2c.py` - Count × probability (analytical & empirical)
- `test_src/validate_cbgig_model_exp2d.py` - Initial 2D attempt (incorrect, kept for reference)
- `test_src/validate_cbgig_model_exp2d_corrected.py` - CbGiG base model validation
- `test_src/validate_cbgig_model_exp2d_fixed.py` - Data leakage check (confirmed no leakage)
- `test_src/validate_gigppw_model_exp2d.py` - GiGpPW base model validation
- `test_src/test_hierarchical_intelligent_exp2e.py` - Intelligent aggregation with actual CbGiG
- `test_src/test_hierarchical_intelligent_exp2e_predicted.py` - Intelligent aggregation with predicted CbGiG
- `test_src/test_focused_composition_exp2g.py` - Focused composition with structural features (SUCCESS)
- `test_src/test_degree_stratified_composition_exp2f.py` - Degree-stratified null model composition

**Results:**
- `results/hierarchical_prediction/experiment1_results.csv`
- `results/hierarchical_prediction/experiment1_features.csv`
- `results/hierarchical_prediction/experiment2a_results.csv`
- `results/hierarchical_prediction/experiment2a_features.csv`
- `results/hierarchical_prediction/exp2a_feature_investigation.csv`
- `results/hierarchical_prediction/experiment2b_optimized_results.csv`
- `results/hierarchical_prediction/experiment2c_results.csv`
- `results/hierarchical_prediction/experiment2d_corrected_summary.csv`
- `results/hierarchical_prediction/experiment2d_gigppw_summary.csv`
- `results/hierarchical_prediction/experiment2d_fixed_summary.csv` (data leakage check)
- `results/hierarchical_prediction/experiment2d_validation_summary.csv` (incorrect attempt)
- `results/hierarchical_prediction/experiment2d_validation_details.csv` (incorrect attempt)
- `results/hierarchical_prediction/experiment2e_results.csv`
- `results/hierarchical_prediction/experiment2e_predicted_results.csv`
- `results/hierarchical_prediction/experiment2e_predicted_breakdown.csv`
- `results/hierarchical_prediction/experiment2f_results.csv`
- `results/hierarchical_prediction/experiment2f_decomposition_analysis.csv`
- `results/hierarchical_prediction/experiment2g_results.csv`
- `results/hierarchical_prediction/experiment2g_feature_importance.csv`

**Visualizations:**
- `results/hierarchical_prediction/experiment1_plots.png`
- `results/hierarchical_prediction/experiment2a_plots.png`
- `results/hierarchical_prediction/experiment2b_optimized_plots.png`
- `results/hierarchical_prediction/experiment2c_plots.png`
- `results/hierarchical_prediction/experiment2d_corrected_plots.png`
- `results/hierarchical_prediction/experiment2d_fixed_plots.png` (data leakage check)
- `results/hierarchical_prediction/experiment2d_gigppw_plots.png`
- `results/hierarchical_prediction/experiment2d_validation_plots.png` (incorrect attempt)
- `results/hierarchical_prediction/experiment2e_plots.png`
- `results/hierarchical_prediction/experiment2e_predicted_plots.png`
- `results/hierarchical_prediction/experiment2g_plots.png`

**Additional Documentation:**
- `docs/2025-11-04_IMPROVING_EXP2E_ANALYSIS.md` - Analysis of improvement strategies
- `docs/2025-11-04_EXPERIMENT_2F_FAILURE_ANALYSIS.md` - Detailed root cause analysis for Exp 2F

### Recommended Next Steps

**For General Pathway Prediction:**
1. **Accept that direct counting is necessary for length-3+ paths**
2. **Focus on optimizing sparse matrix multiplication** for computational efficiency
3. **Investigate non-compositional ML approaches** that use rich features without assuming multiplicative structure
4. **Consider graph neural networks** that can learn complex path patterns
5. **Explore sampling-based methods** for approximating long path counts
6. **Study why length-2 paths behave differently** - what changes at length-3?

**For Null Model Applications:**
1. **Compositional null models may not be viable for 3+ edge pathways**
   - Exp 2F showed degree-stratified composition fails (r=0.60, 55x overprediction)
   - Fundamental tension: composition requires topology, but null models exclude topology
   - Pathway structure is too sparse and specific for degree-only aggregation

2. **Alternative null model strategies:**
   - Direct null models: Train models to predict CbGiGpPW counts directly from degrees (bypassing composition)
   - Pathway enumeration: Enumerate pathways for 1-5 permutations and average (current approach)
   - Hybrid models: Use topology features beyond just degree (e.g., clustering coefficients, hub connectivity)
   - Sparsity-aware sampling: Sample only pathways through high-degree intermediates

3. **For anomaly detection specifically:**
   - May need to enumerate pathways even in permuted graphs
   - Focus on computational optimization rather than compositional approximation
   - Consider pathway-specific models rather than universal composition formulas
