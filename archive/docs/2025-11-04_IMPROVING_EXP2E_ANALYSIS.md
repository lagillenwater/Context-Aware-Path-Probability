# Analysis: Can We Improve Exp 2E to Reach r > 0.95?

Date: 2025-11-04

## Current Exp 2E Performance

**Variant 1 (actual CbGiG counts):**
- Correlation: r = 0.855 (R² = 0.73)
- Systematic underprediction: 10x (mean true: 3.30, mean pred: 0.38)
- Intermediates: 2.5 genes per pair (correct sparsity)
- Formula: `sum over connected genes: CbGiG_count × P_edge_GpPW`

**Why it works better than others:**
- Precise double-filtering (only genes connecting to BOTH C and PW)
- Uses actual topology information (which genes actually connect)
- Maintains sparsity (2.5 intermediates vs 33.8 in Exp 2F)

**Why it falls short of r > 0.95:**
1. 10x systematic underprediction (wrong scale)
2. 27% of variance unexplained (R² = 0.73)
3. Simple linear aggregation (no interaction effects)

## Improvement Strategies

### Strategy 1: Calibration to Fix Scale

**Problem:** Multiplying counts by edge probabilities (0.003-0.02) produces tiny values

**Solution:** Learn a calibration factor

```python
# Training phase
calibration_factor = sum(true × pred_raw) / sum(pred_raw²)

# Test phase
pred_calibrated = calibration_factor × pred_raw
```

**Expected outcome:**
- Fixes 10x underprediction (better MAE)
- Does NOT improve correlation (r stays ~0.855)
- R² remains 0.73

**Verdict:** Helps with scale but doesn't bridge gap to r > 0.95

### Strategy 2: Use Expected CbGiG Instead of Perm 0

**Problem:** Using perm 0 CbGiG counts instead of expected values

**Current formula:**
```python
contrib = CbGiG_perm0[C, G2] × P_edge
```

**Improved formula:**
```python
# Use expected CbGiG (mean of perms 6-20 or predicted)
contrib = E[CbGiG][C, G2] × P_edge
```

**Implementation options:**
a) Use mean CbGiG from perms 6-20 directly
b) Use predicted CbGiG from validated model (r=0.95)

**Challenge:** How to filter false positives?
- If we use predicted CbGiG, we get 47.8 intermediates (Exp 2E-v2: r=0.749)
- Need to filter by actual topology while using expected counts

**Hybrid approach:**
```python
for G2 in genes_connected_to_PW:
    # Filter by actual topology from perm 0
    if CbGiG_perm0[C, G2] == 0:
        continue

    # But use expected count for magnitude
    expected_CbGiG = mean_CbGiG_6_20[C, G2]
    P_edge = empirical_freq[(deg_G2, deg_PW)]
    contrib = expected_CbGiG × P_edge
```

**Expected outcome:**
- Better calibration between training and validation
- May improve both scale and correlation
- Maintains precise filtering (2.5 intermediates)

**Verdict:** Worth testing - could improve r from 0.855 toward 0.90+

### Strategy 3: Rich Feature Composition Model

**Problem:** Simple sum(CbGiG × P_edge) doesn't capture non-linear effects

**Solution:** Train ML model on rich features

**Features for each (C, PW) pair:**
```python
features = [
    # Endpoint degrees
    deg_C, deg_PW, deg_C × deg_PW, deg_C², deg_PW²,

    # Intermediate statistics
    n_intermediates_connected_to_both,
    mean_intermediate_degree,
    max_intermediate_degree,

    # CbGiG aggregates
    sum_CbGiG_counts,
    mean_CbGiG_counts,
    max_CbGiG_count,

    # Edge probability aggregates
    sum_P_edge,
    mean_P_edge,
    max_P_edge,

    # Composition term (from Exp 2E)
    sum(CbGiG × P_edge),

    # Interaction terms
    n_intermediates × mean_CbGiG,
    sum_CbGiG × mean_P_edge,
    # etc.
]

# Train model
model = LinearRegression()  # or RandomForest, NN
model.fit(features_train, true_CbGiGpPW_train)
```

**What this captures:**
- Non-linear interactions between intermediates
- Better weighting of different intermediate contributions
- Systematic biases in the simple formula
- Higher-order degree effects

**Expected outcome:**
- Could capture the missing 27% of variance
- Potential to reach r > 0.90, possibly r > 0.95
- More complex but more flexible

**Verdict:** Most promising approach for reaching r > 0.95

### Strategy 4: Use GpPW Count Model (if applicable)

**Problem:** Using edge probabilities instead of expected counts

**Current:** `P_edge_GpPW` is probability that edge exists (0.003-0.02)

**Alternative:** If GpPW has counts (not just binary edges), use count model

**Need to check:**
- Is GpPW binary (Gene participates or not) or does it have counts?
- From data: 84,372 edges - likely binary participation

**Verdict:** Probably not applicable (GpPW is likely binary)

## Recommended Experiments

### Experiment 2G: Expected CbGiG with Topology Filter

Test whether using expected CbGiG improves over perm 0 counts:

```python
# Compute mean CbGiG from perms 6-20
mean_CbGiG_6_20 = mean([CbGiG_perm_k for k in range(6, 21)])

for C, PW in test_pairs:
    pred = 0
    for G2 in genes_connected_to_PW:
        # Filter by perm 0 topology
        if CbGiG_perm0[C, G2] == 0:
            continue

        # Use expected count
        expected_CbGiG = mean_CbGiG_6_20[C, G2]
        P_edge = empirical_freq[(deg_G2, deg_PW)]
        pred += expected_CbGiG × P_edge
```

**Expected improvement:** r = 0.86-0.90 (modest improvement)

### Experiment 2H: Rich Feature Composition Model

Train ML model with rich features extracted from Exp 2E intermediate results:

```python
# For each pair, extract ~15-20 features
features = extract_composition_features(C, PW, genes_connected_to_PW, ...)

# Train model
model = LinearRegression()  # Start simple, try RF/NN if needed
model.fit(features_train, true_train)

# Validate
r_test = pearsonr(model.predict(features_test), true_test)[0]
```

**Expected improvement:** r = 0.90-0.97 (potentially reaches threshold)

## Feasibility Assessment

**Can we reach r > 0.95?**

**Pessimistic view:**
- Exp 2E achieves r = 0.855 with precise filtering and actual counts
- Missing 27% of variance suggests fundamental limitations
- Even validated base models (r=0.95) fail when composed (r=0.75)
- May be hitting fundamental limits of compositional approaches

**Optimistic view:**
- Exp 2E uses simple linear aggregation
- Base models achieve r > 0.95 for subpaths
- Rich features could capture non-linear interactions
- Calibration + feature engineering could bridge the gap

**Most likely outcome:**
- Calibration alone: r = 0.855 (no change, just fixes scale)
- Expected CbGiG: r = 0.87-0.90 (modest improvement)
- Rich features: r = 0.90-0.94 (significant improvement, may not reach 0.95)
- Combined approaches: r = 0.92-0.96 (possible to reach threshold with effort)

## Recommendation

**Priority 1:** Implement Experiment 2H (rich feature composition model)
- Most promising path to r > 0.95
- Can incorporate lessons from 2E (topology filtering, intermediate statistics)
- Captures non-linear effects that simple aggregation misses

**Priority 2:** If 2H fails, accept r = 0.90-0.93 as practical limit
- Still useful for approximation and understanding
- Better than naive approaches (Exp 2B: r = -0.09)
- May not replace direct enumeration for high-precision needs

**Priority 3:** For null model use case specifically
- Consider direct CbGiGpPW null model (predict 3-edge paths directly from degrees)
- May achieve r > 0.95 without composition
- Avoids compositional limitations entirely
