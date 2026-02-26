# Path Length Analysis: Endpoint-Only Prediction
**Date:** 2025-11-11
**Analysis:** Testing endpoint-only prediction across path lengths 3, 4, and 5

---

## Executive Summary

We tested whether endpoint-only prediction (using only source and target node degrees) extends to longer metapaths. Results reveal a striking pattern:

**Correlation improves with path length, but calibration worsens dramatically.**

| Path Length | Metapath | r | Q-Q | MAE | Pairs with Pathways |
|-------------|----------|---|-----|-----|---------------------|
| 3 (2-hop) | CbGpPW | 0.778 | 0.815 | - | - |
| 4 (3-hop) | CbGiGpPW | 0.828 | 0.730 | 1.904 | 83.4% |
| 5 (4-hop) | CbGiGiGpPW | 0.899 | 0.461 | 104.404 | 94.6% |

**Key finding**: Endpoint degrees become MORE predictive of mean counts as paths lengthen, but residuals become MORE non-normal (heavy tails, extreme outliers).

---

## Methodology

### Test Design

For each path length:
1. Sample 10,000 (Compound, Pathway) pairs
2. Compute pathway counts in perms 0-4 (training) and 15-19 (test)
3. Train Random Forest on 5 endpoint features: [deg_C, deg_PW, deg_C×deg_PW, deg_C², deg_PW²]
4. Evaluate r, Q-Q, and MAE on test permutations

### Metapaths Tested

**Length-3 (CbGpPW)**: Compound→Gene→Pathway
- Edges: CbG @ GpPW
- Baseline from November 11 analysis

**Length-4 (CbGiGpPW)**: Compound→Gene→Gene→Pathway
- Edges: CbG @ GiG @ GpPW
- Adds one intermediate Gene

**Length-5 (CbGiGiGpPW)**: Compound→Gene→Gene→Gene→Pathway
- Edges: CbG @ GiG @ GiG @ GpPW
- Two intermediate Genes

### Model

Random Forest with parameters consistent with November 11 analysis:
- n_estimators=100
- max_depth=10
- min_samples_leaf=5
- random_state=42

---

## Results

### Performance Trends

#### Correlation (r)

**Improves monotonically with path length:**
- Length-3: r = 0.778
- Length-4: r = 0.828 ± 0.012 (+6.4%)
- Length-5: r = 0.899 ± 0.027 (+15.5% from length-3)

**Interpretation**: Endpoint degrees become more informative as paths lengthen. This is counterintuitive but consistent - longer paths have more intermediate possibilities, and high-degree endpoints create combinatorially more paths.

#### Calibration (Q-Q Correlation)

**Degrades dramatically with path length:**
- Length-3: Q-Q = 0.815
- Length-4: Q-Q = 0.730 (-10.4%)
- Length-5: Q-Q = 0.461 (-43.4% from length-3)

**Interpretation**: Residuals become increasingly non-normal. Q-Q plots show severe heavy tails at length-5, with extreme outliers deviating by 3+ standard deviations.

#### Mean Absolute Error (MAE)

**Increases with path length:**
- Length-4: MAE = 1.904 ± 0.059
- Length-5: MAE = 104.404 ± 4.479 (54.8x higher)

**Note**: MAE not directly comparable across lengths due to different count scales:
- Length-4 mean count: 2.853
- Length-5 mean count: 268.964 (94x higher)

Relative MAE (MAE/mean):
- Length-4: 1.904/2.853 = 66.7%
- Length-5: 104.404/268.964 = 38.8%

**Finding**: Relative error actually improves at length-5, consistent with improving r.

### Count Distributions

**Pathway prevalence increases with length:**
- Length-4: 83.4% of pairs have pathways
- Length-5: 94.6% of pairs have pathways

**Count statistics:**
- Length-4: mean=2.853, std=7.028, max=174.6
- Length-5: mean=268.964, std=946.528, max=52,312.6

**Variance increases superlinearly**:
- Length-4 to length-5: mean increases 94x, std increases 135x
- Higher variance reflects combinatorial explosion of paths

### Residual Analysis

**Heteroscedasticity worsens with path length:**

**Length-4**:
- Residuals increase modestly with predicted count
- Most residuals within [-25, +25] for predictions <50
- Some outliers at high counts (residuals up to +100)

**Length-5**:
- Severe heteroscedasticity
- Residuals up to +30,000 for high predictions
- Clear fan pattern: variance scales with predicted count
- Q-Q plot shows dramatic departure at both tails

**Implication**: Uncertainty grows faster than mean count as paths lengthen.

---

## Key Insights

### 1. The Correlation Paradox

**Why does r improve while Q-Q worsens?**

Correlation measures linear relationship strength, not distributional assumptions:
- Endpoint degrees strongly predict MEAN counts (r improves)
- But individual counts have heavy-tailed distributions around the mean (Q-Q worsens)

At length-5, most pairs follow the predicted relationship closely (driving high r), but a few outliers have enormous counts (driving poor Q-Q).

### 2. Combinatorial Pathway Explosion

As paths lengthen:
- Number of possible paths grows combinatorially
- High-degree endpoints create exponentially more paths
- Mean count becomes more predictable (law of large numbers)
- But variance increases due to topology-specific path availability

**Example**: A Compound with degree 50 and Pathway with degree 500:
- Length-3: ~25,000 potential 2-hop paths
- Length-5: ~12.5 million potential 4-hop paths

The mean is predictable from degrees, but specific permutation topology determines which paths exist.

### 3. Topology-Specific Variance Accumulates

Each edge adds topology-specific variance:
- Length-3: 22% unexplained variance (r=0.78)
- Length-4: ~30% unexplained variance (r=0.83)
- Length-5: ~20% unexplained variance (r=0.90) but MUCH heavier tails

**Caveat**: Higher r at length-5 doesn't mean less unexplained variance in absolute terms - it means variance scales predictably with mean (which grows ~100x).

### 4. When to Use This Approach

**Appropriate for:**
- Predicting MEAN pathway counts across permutations
- Ranking pairs by expected connectivity
- Identifying high-degree hubs

**Inappropriate for:**
- Predicting INDIVIDUAL permutation counts (heavy tails)
- Uncertainty quantification without correction (Q-Q=0.461)
- Anomaly detection at extreme values (poor calibration)

---

## Visualizations

### Figure 1: Length-4 Comparison
**File**: `results/length4_endpoint/length3_vs_length4_comparison.png`

**Key observations:**
- Predicted vs Actual: Good linear fit, scatter at high counts
- Residuals: Moderate heteroscedasticity
- Q-Q: Slight heavy tails
- Performance: r=0.81-0.84 across test perms
- Degree distribution: Discrete banding patterns

### Figure 2: Length-5 Comparison
**File**: `results/length5_endpoint/length5_comparison.png`

**Key observations:**
- Predicted vs Actual: Strong linear relationship (r=0.876-0.927)
- Residuals: Severe heteroscedasticity, fan pattern
- Q-Q: Dramatic heavy tails, extreme outliers
- Performance vs Length: Bar chart showing r increases, Q-Q decreases
- Degree distribution: Counts scale exponentially with degree product

---

## Statistical Implications

### Law of Large Numbers Effect

As paths lengthen, each pair contains more paths:
- Averaging over many paths smooths out topology-specific noise
- Mean becomes predictable from degrees (central limit theorem)
- But extreme values in the tail become MORE extreme (not less)

### Why Q-Q Degrades

The residual distribution transitions from approximately normal to heavy-tailed:
- Length-3: Moderate tails
- Length-4: Heavy tails
- Length-5: Extreme tails with outliers >4 SD

**Cause**: Multiplicative path counting. A few pairs have topology that creates exceptional connectivity, leading to counts 10-100x above predicted mean.

### Implications for Modeling

Standard approaches assume:
- Normally distributed residuals (violated)
- Homoscedastic errors (violated)
- No extreme outliers (violated)

**Recommendations**:
1. Use robust regression methods (e.g., Huber loss)
2. Model variance separately (heteroscedastic models)
3. Log-transform counts before modeling
4. Use quantile regression for extreme values

---

## Answer to Original Question

**Q: Will this approach extend to longer paths? How feasible would that be?**

**A: Yes, but with important caveats.**

### Feasibility Assessment

**Good news:**
- Endpoint-only prediction WORKS for longer paths
- Correlation actually IMPROVES (0.78 → 0.83 → 0.90)
- Computationally feasible (minutes for length-5)
- No need for intermediate node features

**Bad news:**
- Calibration DEGRADES severely (0.82 → 0.73 → 0.46)
- Extreme outliers become unpredictable
- Uncertainty quantification requires special handling
- Residuals are non-normal with heavy tails

### Practical Limits

**Length-5 appears near the practical limit:**
- Q-Q = 0.461 means poor distributional calibration
- Outliers up to 30,000 when mean is 269
- Would need log-transform or quantile methods

**Length-6+ predictions:**
- Correlation may continue improving
- But calibration likely continues degrading
- Computational cost increases (5+ matrix multiplications)
- Practical utility questionable with Q-Q < 0.4

### Recommendations

**For mean prediction (ranking, scoring):**
- Use endpoint-only approach up to length-6
- Expect r > 0.80 performance
- Random Forest works well

**For uncertainty quantification:**
- Length-3: Use standard methods (Q-Q=0.82)
- Length-4: Use heteroscedastic models (Q-Q=0.73)
- Length-5+: Require specialized approaches (quantile regression, log-normal models)

**For anomaly detection on Hetionet:**
- Length-3: Recommend (well-calibrated)
- Length-4: Use with caution (moderate calibration)
- Length-5+: Not recommended (poor calibration)

---

## Conclusions

### Main Findings

1. **Endpoint-only prediction generalizes to longer paths** with improving correlation
2. **Calibration degrades** as residuals become increasingly heavy-tailed
3. **Law of large numbers** makes means predictable but extremes more extreme
4. **Practical limit** around length-5 due to calibration breakdown

### Why This Happens

Longer paths have:
- More combinatorial possibilities (predictable from degrees)
- More topology-specific variance (heavy tails)
- Multiplicative counting (extreme outliers)

Endpoint degrees capture the mean well (law of large numbers) but miss topology-specific path availability that creates outliers.

### Implications for Null Models

The degree-based null model:
- Successfully predicts EXPECTED pathway counts
- Captures how degree drives connectivity
- But cannot predict SPECIFIC permutation topology

This is exactly what a null model should do - provide degree-based expectations while acknowledging irreducible topological variance.

### Future Work

1. Test on other metapaths (CtDaGiG, etc.) to confirm generalization
2. Develop log-normal or quantile regression models for calibration
3. Quantify relationship between path length and Q-Q degradation
4. Test length-6 to confirm practical limit
5. Compare to compositional approaches (multiplicative vs additive)

---

## Files Generated

### Scripts
- `test_src/test_length4_endpoint_only.py` - Length-4 analysis
- `test_src/test_length5_endpoint_only.py` - Length-5 analysis

### Results
- `results/length4_endpoint/CbGiGpPW_endpoint_results.csv`
- `results/length5_endpoint/CbGiGiGpPW_endpoint_results.csv`

### Visualizations
- `results/length4_endpoint/length3_vs_length4_comparison.png`
- `results/length5_endpoint/length5_comparison.png`

### Documentation
- `docs/2025-11-11_PATH_LENGTH_ANALYSIS.md` (this file)

---

**Analysis completed:** 2025-11-11
**Key achievement:** Demonstrated endpoint-only prediction generalizes to longer paths with r>0.80, but requires specialized uncertainty quantification methods
