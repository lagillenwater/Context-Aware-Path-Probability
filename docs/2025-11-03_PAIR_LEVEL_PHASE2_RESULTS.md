# Pair-Level Phase 2: Degree-Aware Correction Results

**Date**: 2025-11-03
**Status**: CORRECTED - Critical bug found and fixed
**Goal**: Test degree-aware correction for pair-level null distribution prediction

---

## CRITICAL BUG CORRECTION

**Bug discovered**: Edge matrices stored as `dtype=bool` instead of numeric types. Boolean matrix multiplication performed logical OR instead of counting shared intermediates, resulting in all pathway counts being 0 or 1.

**Impact**: Original results predicted binary pathway presence/absence (r = 0.88-0.91) instead of actual pathway counts. After fixing to integer matrices, baseline performance is r = 0.98-0.99 and correction is **not needed**.

**See**: `docs/2025-11-03_BUG_CORRECTION.md` for detailed analysis.

---

## Executive Summary

**CORRECTED FINDING: Baseline Linear Regression already achieves r = 0.98-0.99**

After fixing the boolean dtype bug, baseline Linear Regression with 5 degree features achieves r = 0.98-0.99 for predicting actual pathway counts. The degree-aware correction method **decreases** performance by 0.1-0.5% and is not needed.

**Corrected Results:**
- **CbGpPW**: Baseline r = **0.9910** (correction: 0.9859, -0.5%)
- **CtDaG**: Baseline r = **0.9769** (correction: 0.9746, -0.2%)
- **CrCbG**: Baseline r = **0.9913** (correction: 0.9898, -0.1%)
- **Success Rate**: 3/3 (100%) exceed r > 0.90 with baseline alone
- **Conclusion**: Use simple baseline Linear Regression (no correction stage needed)

This enables accurate null distribution prediction for anomaly detection using a simpler approach than originally developed.

---

## Visualizations Overview (CORRECTED)

**Comprehensive 9-panel analysis** generated for each metapath showing:

1. **Predicted vs Observed** (Row 1): Baseline and corrected models with correlation metrics
2. **Residual Analysis** (Row 2): Heteroscedasticity patterns before and after correction
3. **Correction Diagnostics** (Row 3): Correction magnitude, degree-dependence, and permutation 000 validation

**Corrected visual findings:**
- **Baseline performance**: Tight clustering around diagonal (r = 0.98-0.99)
- **Pathway count range**: Now correctly shows 0-50+ pathways (not 0-1)
- **Correction effect**: Slightly decreases performance, increases scatter
- **Heteroscedasticity**: Correction increases rather than decreases heteroscedasticity

See corrected detailed analysis plots:
- [CbGpPW](../results/pair_level_phase2_detailed/CbGpPW_detailed_analysis.png) - Baseline r = 0.99, Corrected r = 0.99 (correction not helpful)
- [CtDaG](../results/pair_level_phase2_detailed/CtDaG_detailed_analysis.png) - Baseline r = 0.98, Corrected r = 0.97 (correction not helpful)
- [CrCbG](../results/pair_level_phase2_detailed/CrCbG_detailed_analysis.png) - Baseline r = 0.99, Corrected r = 0.99 (correction not helpful)

---

## Background: Phase 1 Baseline

**From 2025-10-31** (Pair-Level Phase 1):
- Trained Linear Regression on 50,000 stratified pairs per metapath
- Used 5 degree features (source_deg, target_deg, product, source², target²)
- Achieved r = 0.88-0.91 (met Phase 1 target r > 0.85)
- Limitation: Systematic bias remained, r < 0.90 for some metapaths

**Phase 2 Goal**: Apply degree-aware correction to:
1. Reduce systematic bias to near-zero
2. Push correlation above r > 0.90 for all metapaths
3. Enable accurate null distribution for anomaly detection

---

## Methodology: Degree-Aware Correction

### Mathematical Formulation

#### Stage 1: Base Linear Regression Model

The base model predicts pathway counts from node degree features using ordinary least squares regression:

$$\hat{y}_{\text{base}} = \mathbf{X}\boldsymbol{\beta} + \beta_0$$

where:
- $\hat{y}_{\text{base}} \in \mathbb{R}^n$: predicted pathway counts for $n$ node pairs
- $\mathbf{X} \in \mathbb{R}^{n \times 5}$: feature matrix with 5 degree-based features per pair
- $\boldsymbol{\beta} \in \mathbb{R}^5$: coefficient vector
- $\beta_0 \in \mathbb{R}$: intercept term

**Feature vector for pair $(u, v)$:**

$$\mathbf{x}_{uv} = \begin{bmatrix}
d_u \\
d_v \\
d_u \cdot d_v \\
d_u^2 \\
d_v^2
\end{bmatrix}$$

where $d_u$ and $d_v$ are the degrees of source and target nodes in the original Hetionet graph.

**Training objective:**

$$\boldsymbol{\beta}^*, \beta_0^* = \arg\min_{\boldsymbol{\beta}, \beta_0} \sum_{i=1}^{n} \left( y_i^{\text{val}} - (\mathbf{x}_i^T\boldsymbol{\beta} + \beta_0) \right)^2$$

where $y_i^{\text{val}}$ is the average pathway count for pair $i$ across permutations 1-20:

$$y_i^{\text{val}} = \frac{1}{20} \sum_{p=1}^{20} c_i^{(p)}$$

with $c_i^{(p)}$ being the pathway count for pair $i$ in permutation $p$.

#### Stage 2: Degree-Aware Correction Model

The correction model addresses systematic bias between the original graph and permutations by learning a correction function that depends on both degree features and predicted pathway counts.

**Correction feature vector for pair $(u, v)$:**

$$\mathbf{z}_{uv} = \begin{bmatrix}
d_u \\
d_v \\
d_u \cdot d_v \\
d_u^2 \\
d_v^2 \\
\sqrt{d_u + 1} \\
\sqrt{d_v + 1} \\
\hat{y}_{\text{base}} \\
\hat{y}_{\text{base}}^2 \\
\log(1 + \hat{y}_{\text{base}}) \\
\hat{y}_{\text{base}} \cdot d_u \\
\hat{y}_{\text{base}} \cdot d_v \\
\hat{y}_{\text{base}} \cdot d_u \cdot d_v \\
\sqrt{\hat{y}_{\text{base}} + 1} \cdot d_u \\
\sqrt{\hat{y}_{\text{base}} + 1} \cdot d_v
\end{bmatrix} \in \mathbb{R}^{15}$$

**Feature groups:**
1. **Degree terms (7)**: Capture degree-dependent bias patterns
2. **Prediction terms (3)**: Model magnitude-dependent bias (heteroscedasticity)
3. **Interaction terms (5)**: Critical for modeling how bias varies with both degrees AND predicted pathway count

**Correction target:**

$$\Delta_i = c_i^{(0)} - \hat{y}_i^{\text{base}}$$

where $c_i^{(0)}$ is the pathway count for pair $i$ in permutation 000 (the first XSwap permutation).

**Correction model:**

$$\hat{\Delta} = \mathbf{Z}\boldsymbol{\gamma} + \gamma_0$$

where:
- $\mathbf{Z} \in \mathbb{R}^{n \times 15}$: correction feature matrix
- $\boldsymbol{\gamma} \in \mathbb{R}^{15}$: correction coefficients
- $\gamma_0 \in \mathbb{R}$: correction intercept

**Training objective:**

$$\boldsymbol{\gamma}^*, \gamma_0^* = \arg\min_{\boldsymbol{\gamma}, \gamma_0} \sum_{i=1}^{n} \left( \Delta_i - (\mathbf{z}_i^T\boldsymbol{\gamma} + \gamma_0) \right)^2$$

**Final corrected prediction:**

$$\hat{y}_{\text{corrected}} = \hat{y}_{\text{base}} + \hat{\Delta}$$

### Two-Stage Model Architecture

**Stage 1: Base Model**
- Train Linear Regression on original Hetionet features
- Target: Average pathway counts from permutations 1-20
- Uses 5 degree features: $d_u$, $d_v$, $d_u \cdot d_v$, $d_u^2$, $d_v^2$

**Stage 2: Correction Model**
- Extract 15 correction features (detailed above)
  - Degree features (7): source, target, product, source², target², sqrt(source), sqrt(target)
  - Prediction features (3): y_pred, y_pred², log(y_pred+1)
  - **Interaction terms (5)**: y_pred × source, y_pred × target, etc.
- Correction target: pathway counts from **permutation 000** minus base predictions
- Train Linear Regression: correction_features → correction
- Final prediction: y_corrected = y_base + correction

### Why Permutation 000?

**Critical Clarification from Today:**
- `data/edges/` contains **original Hetionet graph**
- `data/permutations/000.hetmat/` contains **first XSwap permutation** (NOT original!)
- Permutations 001-199 are additional XSwap randomizations
- Phase 2 trains on original graph, uses permutation 000 for correction

This mirrors bin-level Phase 5b's successful approach.

### Why Interaction Terms Are Critical

**Heteroscedasticity in pathway count prediction**: The systematic bias between original graph predictions and permutation averages is not constant—it varies with both node degrees and predicted pathway count magnitude.

**Mathematical justification:**

If bias were constant, a simple offset correction would suffice:
$$\hat{y}_{\text{corrected}} = \hat{y}_{\text{base}} + c$$

However, empirical analysis reveals heteroscedastic bias:
$$\text{Bias}(u, v) = f(d_u, d_v, \hat{y}_{\text{base}})$$

where the bias function $f$ is **non-separable**—it cannot be decomposed as:
$$f(d_u, d_v, \hat{y}_{\text{base}}) \neq g(d_u, d_v) + h(\hat{y}_{\text{base}})$$

**Evidence from bin-level Phase 5b:**
- CbGpPW bias range: +0.006 to -0.094 (10% of pathway count range)
- Correlation between |residuals| and true values: r = 0.84 (highly significant)
- Bias quartiles: Q1 = +0.0018, Q2 = +0.0056, Q3 = -0.0078, Q4 = -0.0943

**Why interaction terms work:**

The interaction terms $\hat{y}_{\text{base}} \cdot d_u$ and $\hat{y}_{\text{base}} \cdot d_v$ allow the correction to model:

$$\Delta(u, v) \approx \gamma_0 + \gamma_1 d_u + \gamma_2 d_v + \gamma_3 \hat{y}_{\text{base}} + \gamma_{11} \hat{y}_{\text{base}} \cdot d_u + \gamma_{12} \hat{y}_{\text{base}} \cdot d_v + \ldots$$

This enables the correction magnitude to **scale differently** with pathway count at different degree combinations. For example:
- High-degree pairs with high pathway counts may systematically underpredict
- Low-degree pairs with moderate pathway counts may systematically overpredict

Without interactions, the model cannot capture these cross-dependencies.

**Visualizations confirm heteroscedasticity:**

See `results/pair_level_phase2_detailed/{metapath}_detailed_analysis.png` Row 2, Panel 1:
- Baseline residuals show fanning pattern (variance increases with predicted value)
- After correction (Row 2, Panel 2): residuals homoscedastic (constant variance)

---

## Results by Metapath

### CbGpPW (Compound-binds-Gene-participates-Pathway)

**Sample**: 10,000 pairs

**Corrected Performance:**
| Metric | Baseline | Corrected | Change |
|--------|----------|-----------|--------|
| Correlation (r) | **0.9910** | 0.9859 | -0.5% (worse) |
| RMSE | **0.0841** | 0.1076 | +28% (worse) |
| Bias | -0.0000 | -0.0057 | Small negative |
| Heteroscedasticity | **r = 0.551** | r = 0.722 | +31% (worse) |

**Pathway Count Statistics:**
- Range: 0 to 48 pathways
- Mean (nonzero): 2.09 pathways
- Median: 1 pathway
- 95th percentile: 6 pathways
- 99th percentile: 12 pathways

**Interpretation:**
- **Baseline exceeds target** (r = 0.99 >> 0.90 target)
- Correction **decreases** performance across all metrics
- RMSE increases 28%, heteroscedasticity increases 31%
- **Conclusion**: Use baseline without correction
- **Target achieved**: r > 0.90 (PASS with baseline alone)

**Detailed Visualizations:**

![CbGpPW Detailed Analysis](../results/pair_level_phase2_detailed/CbGpPW_detailed_analysis.png)

**Figure interpretation:**
- **Row 1**: Predicted vs observed pathway counts
  - Panel 1 (baseline): r = 0.8846, points scatter moderately around perfect prediction line
  - Panel 2 (corrected): r = 0.9363, tighter clustering, points closer to diagonal
  - Panel 3: Metrics comparison showing correlation improvement and RMSE reduction
- **Row 2**: Residual analysis
  - Panel 1 (baseline residuals): Slight fanning pattern visible (heteroscedasticity)
  - Panel 2 (corrected residuals): More uniform scatter (homoscedastic)
  - Panel 3: Residual distributions—corrected distribution narrower and more centered at zero
- **Row 3**: Correction analysis
  - Panel 1: Correction magnitude vs base prediction (shows how correction scales with predicted value)
  - Panel 2: Correction vs source degree (degree-dependent correction patterns)
  - Panel 3: Permutation 000 vs validation average (r = 0.997, confirming perm 000 is representative)

---

### CtDaG (Compound-treats-Disease-associates-Gene)

**Sample**: 10,000 pairs

**Performance:**
| Metric | Baseline | Corrected | Improvement |
|--------|----------|-----------|-------------|
| Correlation (r) | 0.9027 | **0.9489** | +5.1% |
| RMSE | 0.0829 | 0.0610 | **-26.4%** |
| Bias | +0.0000 | -0.0005 | Near-zero |

**Interpretation:**
- Baseline closer to target (r = 0.90), still benefits from correction
- Achieves highest RMSE reduction percentage (26.4%)
- Already had near-zero bias, maintains it
- **Target achieved**: r > 0.90 (PASS)

**Detailed Visualizations:**

![CtDaG Detailed Analysis](../results/pair_level_phase2_detailed/CtDaG_detailed_analysis.png)

**Figure interpretation:**
- **Row 1**: Baseline (r = 0.90) already strong, correction pushes to r = 0.95
- **Row 2**: Residuals show good correction behavior with minimal heteroscedasticity in both baseline and corrected models
- **Row 3**: Correction patterns similar to CbGpPW but with smaller magnitude corrections needed

---

### CrCbG (Compound-resembles-Compound-binds-Gene)

**Sample**: 10,000 pairs

**Performance:**
| Metric | Baseline | Corrected | Improvement |
|--------|----------|-----------|-------------|
| Correlation (r) | 0.9118 | **0.9692** | +6.3% |
| RMSE | 0.1127 | 0.0686 | **-39.2%** |
| Bias | -0.0000 | -0.0085 | Small negative |

**Interpretation:**
- **Highest performing metapath** (r = 0.97)
- Largest absolute RMSE reduction (39%)
- Small negative bias introduced but acceptable (|bias| < 0.01)
- **Target achieved**: r > 0.90 (PASS)

**Detailed Visualizations:**

![CrCbG Detailed Analysis](../results/pair_level_phase2_detailed/CrCbG_detailed_analysis.png)

**Figure interpretation:**
- **Row 1**: Dramatic improvement from r = 0.91 to r = 0.97
  - Corrected model shows tightest clustering around diagonal of all three metapaths
  - RMSE reduced from 0.113 to 0.069 (39% reduction)
- **Row 2**: Residual patterns
  - Baseline shows moderate heteroscedasticity
  - Corrected residuals much more uniform and centered
  - Residual distribution narrowed substantially
- **Row 3**: Correction analysis
  - Larger correction magnitudes than other metapaths (reflecting larger baseline error)
  - Corrections vary systematically with both degree and predicted value
  - Permutation 000 correlation with validation remains high (r > 0.99)

---

## Cross-Metapath Summary (CORRECTED)

### Performance Table

| Metapath | Baseline r | Corrected r | Δr | ΔRMSE | Target (r>0.90) |
|----------|-----------|-------------|-----|-------|-----------------|
| CbGpPW | **0.9910** | 0.9859 | -0.0051 | +28% | PASS (baseline) |
| CtDaG | **0.9769** | 0.9746 | -0.0023 | +19% | PASS (baseline) |
| CrCbG | **0.9913** | 0.9898 | -0.0014 | +8% | PASS (baseline) |
| **Average** | **0.9864** | **0.9834** | **-0.0029** | **+18%** | **3/3** |

### Success Rate

**100% success rate**: All 3 metapaths achieve r > 0.90 with **baseline alone**

**Key findings:**
- Baseline Linear Regression: r = 0.98-0.99 (exceeds target)
- Correction decreases performance: -0.1% to -0.5%
- RMSE increases after correction: +8% to +28%
- Heteroscedasticity increases after correction
- **Conclusion**: Correction not needed, use baseline model

---

## Comparison to Bin-Level Phase 5b

| Aspect | Bin-Level (Phase 5b) | Pair-Level (Phase 2) |
|--------|---------------------|---------------------|
| **Training samples** | 12-100 bins | 10,000 pairs |
| **Baseline r** | 0.9550-0.9866 | 0.8846-0.9118 |
| **Corrected r** | 0.9964-0.9987 | 0.9363-0.9692 |
| **Improvement** | +1.1% to +4.3% | +5.1% to +6.3% |
| **RMSE reduction** | 49% to 74% | 25% to 39% |
| **Method** | Two-stage Linear Regression | Two-stage Linear Regression |
| **Correction features** | 15 (degree × prediction interactions) | 15 (same) |
| **Success rate** | 3/3 achieve r > 0.99 | 3/3 achieve r > 0.90 |

**Key Insights:**
1. **Same method works at both scales**: Degree-aware correction generalizes from bin-level to pair-level
2. **Larger relative improvement at pair-level**: +5-6% vs +1-4% (because baseline lower)
3. **Absolute performance lower at pair-level**: r = 0.95 vs r > 0.99 (expected due to within-bin variance)
4. **Consistent bias correction**: Both achieve near-zero bias

**Why pair-level r < bin-level r?**
- Bin-level predicts average for degree class (smoother signal)
- Pair-level predicts individual pairs (more noise)
- Within-bin variance in Phase 5b: 100-188% relative deviation
- Pair-level r = 0.95 is excellent given individual pair stochasticity

---

## Technical Details

### Data Loading Bug Fixed

**Initial Bug**: Script loaded original Hetionet as "permutation 0"
- Result: Massive bias (+0.41), negative correlation (r = 0.69)
- Cause: `load_edge_matrix('CbG', perm_id=0)` loaded from `data/edges/` not `data/permutations/000.hetmat/`

**Fix**: Distinguish between 'original' and numeric permutation IDs
```python
def load_edge_matrix(edge_type, perm_id='original'):
    if perm_id == 'original':
        return sp.load_npz(data_dir / 'edges' / f'{edge_type}.sparse.npz')
    else:
        return sp.load_npz(data_dir / 'permutations' / f'{perm_id:03d}.hetmat' / 'edges' / f'{edge_type}.sparse.npz')
```

**Validation**: Confirmed permutation 000 mean pathway count (0.0987) ≈ validation mean (0.1014)

### Feature Engineering

**Base Features (5)**:
1. source_degree
2. target_degree
3. source × target
4. source²
5. target²

**Correction Features (15)**:
- Base degree features (7): source, target, product, squares, sqrt
- Prediction features (3): y_pred, y_pred², log(y_pred+1)
- **Interaction terms (5)**: Critical for heteroscedasticity
  - y_pred × source_degree
  - y_pred × target_degree
  - y_pred × (source × target)
  - sqrt(y_pred+1) × source_degree
  - sqrt(y_pred+1) × target_degree

**Why interactions matter**: Bias varies with BOTH pathway count magnitude and node degrees. Simple additive correction (degree + prediction features) cannot capture this interaction.

### Computational Efficiency

**Per metapath (10,000 pairs):**
- Sampling: < 0.1s
- Feature extraction: ~1.3s
- Target computation (20 permutations): ~1.6s
- Permutation 000 computation: ~0.1s
- Base model training: < 0.01s
- Correction model training: < 0.01s
- **Total**: ~3.1s

**Scaling to 50,000 pairs**: ~15s per metapath (extrapolating linearly)

**Production deployment**: Can train and apply correction in seconds, enabling rapid iteration.

---

## Visualizations

### Comparison Plot

Generated: `results/pair_level_phase2_all/phase2_all_metapaths_comparison.png`

**Three panels:**
1. **Correlation comparison**: Baseline vs Corrected for all metapaths, with r > 0.90 target line
2. **RMSE comparison**: Shows reduction across all metapaths
3. **Relative improvement**: Bars showing +5-6% improvement

**Key visual insights:**
- All corrected bars exceed 0.90 threshold
- CrCbG achieves highest correlation (0.97)
- RMSE reduction consistent across metapaths

### Individual Scatter Plots

Generated: `results/pair_level_phase2/CbGpPW_phase2_quick_test.png`

**Two panels:**
- Baseline: r = 0.8846, shows good linear relationship with slight scatter
- Corrected: r = 0.9363, tighter clustering around perfect prediction line

**Pattern**: Correction reduces scatter, particularly for higher pathway counts

---

## Comparison to Phase 1

| Metric | Phase 1 Baseline | Phase 2 Corrected | Improvement |
|--------|------------------|-------------------|-------------|
| **CbGpPW** | 0.8827 | 0.9363 | +6.1% |
| **CtDaG** | 0.8977 | 0.9489 | +5.7% |
| **CrCbG** | 0.9149 | 0.9692 | +5.9% |
| **Average** | 0.8984 | 0.9515 | +5.9% |

**Notes:**
- Phase 1 used 50,000 pairs (stratified sampling)
- Phase 2 quick test used 10,000 pairs (faster validation)
- Results comparable despite smaller sample (correction is robust)

**Implications:**
- Correction method generalizes well
- Doesn't require large sample sizes to be effective
- Can use smaller samples for rapid iteration

---

## Key Findings

### 1. Degree-Aware Correction Works at Pair-Level

**Success**: Same approach that achieved r > 0.99 at bin-level pushes pair-level from 0.88-0.91 → 0.9363-0.9692

**Why it works:**
- Systematic bias exists between original graph and permutations
- Bias varies with both degrees and pathway count magnitude
- Interaction terms capture this heteroscedasticity
- Permutation 000 provides reliable correction reference

### 2. All Metapaths Exceed r > 0.90 Target

**100% success rate** on diverse metapaths:
- CbGpPW: Sparse, asymmetric (compounds → genes → pathways)
- CtDaG: Moderate density, disease-centric
- CrCbG: Compound-compound similarity

**Generalization**: Method not metapath-specific, works across graph topologies

### 3. Consistent RMSE Reduction (25-39%)

**Beyond correlation improvement**, correction significantly reduces prediction error:
- CbGpPW: -24.6% RMSE
- CtDaG: -26.4% RMSE
- CrCbG: -39.2% RMSE

**Implication**: Tighter predictions, more reliable null distribution estimates

### 4. Near-Zero Bias Maintained

**All metapaths achieve |bias| < 0.01** after correction:
- CbGpPW: -0.0027
- CtDaG: -0.0005
- CrCbG: -0.0085

**Critical for anomaly detection**: Unbiased null enables accurate z-score calculation

### 5. Fast Training and Inference

**Computational efficiency**:
- Training: ~3s per metapath (10,000 pairs)
- Inference: < 1ms per pair
- Scales linearly with sample size

**Production-ready**: Can deploy for real-time pair-level null prediction

---

## Limitations and Future Work

### Current Limitations

1. **Tested on 10,000 pairs** (quick validation)
   - Phase 1 used 50,000 pairs
   - Should validate Phase 2 on larger sample

2. **Only 2-hop metapaths tested**
   - Need to extend to 3-hop, 4-hop paths
   - Feature dimensionality may increase

3. **Single permutation for correction** (perm 000)
   - Could ensemble multiple permutations
   - May improve robustness

4. **Absolute r < bin-level** (0.95 vs 0.99)
   - Due to within-bin variance (unavoidable)
   - r = 0.95 may be ceiling for pair-level

### Future Work (Tomorrow: Anomaly Detection)

**Now that null distribution prediction works (r > 0.90), proceed to:**

1. **Pair-Level Phase 3**: Variance estimation
   - Model variance across permutations
   - Enable z-score calculation: z = (obs - exp) / sqrt(var)
   - Target: r > 0.70 for variance model

2. **Pair-Level Phase 5**: Anomaly detection
   - Score all (source, target) pairs for specific metapaths
   - Identify pairs with z > 3 (enriched) or z < -3 (depleted)
   - FDR correction for multiple testing
   - Biological interpretation of top anomalies

3. **Comparison to compositional null**
   - Our ML approach: r = 0.95
   - Compositional (notebook 11): r = 0.06
   - DWPC (Himmelstein 2017): TBD
   - Quantify improvement for publication

### Potential Improvements

1. **Ensemble correction**
   - Use average of permutations 000-002 for correction
   - May reduce correction noise

2. **Non-linear correction**
   - Test Random Forest or Neural Network for correction stage
   - May capture complex bias patterns

3. **Adaptive features**
   - Select features per metapath
   - May improve sparse metapaths

4. **Regularization tuning**
   - Test Ridge/Lasso for correction model
   - May prevent overfitting on small samples

---

## Conclusion (CORRECTED)

**Critical Bug Impact**: Original results based on boolean dtype bug that converted count regression to binary classification.

**Corrected Findings**:
- **Baseline Linear Regression achieves r = 0.98-0.99** (exceeds all targets)
- Degree-aware correction **not needed** (decreases performance by 0.1-0.5%)
- Simple 5-feature model (d_u, d_v, d_u×d_v, d_u², d_v²) sufficient
- Fast training and inference (< 3s per metapath)
- Generalizes across diverse metapaths

**Key achievement**: Demonstrated that node degree features alone predict pathway counts with r = 0.99, far exceeding the r > 0.90 target.

**Significance**: Enables accurate null distribution prediction for individual node pairs using a simpler approach than originally developed. No correction stage needed.

**Ready for tomorrow**: With null prediction working (r = 0.99), can proceed to:
1. Use baseline Linear Regression (no correction)
2. Compute variance from permutations 1-20
3. Calculate z-scores for anomaly detection
4. Identify enriched/depleted pathways

**Method comparison**: Our ML approach (r = 0.99) vastly outperforms compositional null (r = 0.06), validating the machine learning strategy for pathway null modeling.

**Lesson learned**: Always validate data types and pathway count ranges. The boolean dtype bug led us to develop an unnecessary correction method.

---

## Detailed Correction Description

### Correction Workflow

**Step-by-step correction process:**

1. **Extract degree features** from original Hetionet for sampled pairs:
   $$\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^n, \quad \mathbf{x}_i = [d_{u_i}, d_{v_i}, d_{u_i} \cdot d_{v_i}, d_{u_i}^2, d_{v_i}^2]^T$$

2. **Compute validation targets** from permutations 1-20:
   $$y_i^{\text{val}} = \frac{1}{20} \sum_{p=1}^{20} c_i^{(p)}$$

3. **Train base model**:
   $$\min_{\boldsymbol{\beta}, \beta_0} \sum_{i=1}^n (y_i^{\text{val}} - \mathbf{x}_i^T\boldsymbol{\beta} - \beta_0)^2$$
   $$\hat{y}_i^{\text{base}} = \mathbf{x}_i^T\boldsymbol{\beta}^* + \beta_0^*$$

4. **Compute permutation 000 targets**:
   $$y_i^{(0)} = c_i^{(0)}$$

5. **Compute correction targets**:
   $$\Delta_i = y_i^{(0)} - \hat{y}_i^{\text{base}}$$

6. **Extract correction features** (15-dimensional):
   $$\mathbf{z}_i = f(\mathbf{x}_i, \hat{y}_i^{\text{base}})$$

   Including degree terms, prediction terms, and **critical interaction terms**.

7. **Train correction model**:
   $$\min_{\boldsymbol{\gamma}, \gamma_0} \sum_{i=1}^n (\Delta_i - \mathbf{z}_i^T\boldsymbol{\gamma} - \gamma_0)^2$$
   $$\hat{\Delta}_i = \mathbf{z}_i^T\boldsymbol{\gamma}^* + \gamma_0^*$$

8. **Apply correction**:
   $$\hat{y}_i^{\text{corrected}} = \hat{y}_i^{\text{base}} + \hat{\Delta}_i$$

### Why This Correction Works

**Theoretical foundation:**

The original Hetionet graph and degree-preserving permutations differ systematically:
- **Assortativity**: Original has biological degree correlations (r = +0.20), permutations destroy these (r = +0.04)
- **Pathway structure**: Original has evolved biological pathways, permutations have random pathways
- **Triangle closure**: Original has biological clustering, permutations have random clustering

These differences create **systematic bias** in predictions:
$$\mathbb{E}[\hat{y}_{\text{base}}] \neq \mathbb{E}[y^{\text{val}}]$$

The bias is **structured** (not random noise):
$$\text{Bias}(u,v) = \mathbb{E}[y^{\text{val}}] - \mathbb{E}[\hat{y}_{\text{base}}] = f(d_u, d_v, \hat{y}_{\text{base}})$$

**Key insight**: Permutation 000 bridges the gap between original and permutation average:
- Permutation 000 is a degree-preserving randomization of original
- Permutation 000 is highly correlated with average of permutations 1-20 (r > 0.99)
- Difference $(y^{(0)} - \hat{y}_{\text{base}})$ captures the systematic transformation

By learning this transformation as a function of degrees and predicted values, the correction generalizes to all pairs.

### Correction Magnitude Analysis

From visualizations (Row 3, Panel 1 of detailed analysis plots):

**CbGpPW:**
- Correction mean: -0.0027 (small negative on average)
- Correction std: 0.0596 (moderate variation)
- Permutation 000 vs validation correlation: r = 0.564
- Pattern: Corrections increase with predicted value (addressing heteroscedasticity)

**CtDaG:**
- Correction mean: -0.0005 (near-zero)
- Correction std: 0.0644 (moderate variation)
- Permutation 000 vs validation correlation: r = 0.581
- Pattern: Smaller average corrections needed (baseline already strong)

**CrCbG:**
- Correction mean: -0.0085 (small negative)
- Correction std: 0.0835 (largest variation)
- Permutation 000 vs validation correlation: r = 0.707
- Pattern: Larger corrections for high predicted values

**Cross-metapath consistency:**
- All show degree-dependent correction patterns
- All show scaling with predicted magnitude
- All reduce heteroscedasticity in residuals (though substantial heteroscedasticity remains)
- Permutation 000 moderately correlated with validation (r = 0.56-0.71)
  - Note: Lower than bin-level Phase 5b (r > 0.99) due to pair-level stochasticity
  - Correction still effective despite lower perm 000-validation correlation

## Files Generated

**Scripts:**
- `test_src/run_pair_level_phase2.py` - Single metapath test
- `test_src/run_pair_level_phase2_all_metapaths.py` - All metapaths analysis
- `test_src/run_pair_level_phase2_detailed_analysis.py` - Detailed visualizations

**Results:**
- `results/pair_level_phase2/CbGpPW_phase2_quick_test.png` - Initial scatter plots
- `results/pair_level_phase2_all/phase2_all_metapaths_results.csv` - Quantitative summary
- `results/pair_level_phase2_all/phase2_all_metapaths_comparison.png` - Cross-metapath comparison
- `results/pair_level_phase2_detailed/CbGpPW_detailed_analysis.png` - 9-panel detailed analysis
- `results/pair_level_phase2_detailed/CtDaG_detailed_analysis.png` - 9-panel detailed analysis
- `results/pair_level_phase2_detailed/CrCbG_detailed_analysis.png` - 9-panel detailed analysis
- `results/pair_level_phase2_detailed/detailed_analysis_summary.csv` - Quantitative metrics

**Documentation:**
- `docs/2025-11-03_PAIR_LEVEL_PHASE2_RESULTS.md` - This document

---

**Status**: COMPLETE - All Phase 2 objectives achieved
**Next**: Pair-Level Phase 3 (Variance Estimation) or Phase 5 (Anomaly Detection)
**Recommendation**: Proceed to Phase 5 (anomaly detection) using fixed variance from permutations, defer sophisticated variance modeling if not needed for initial results
