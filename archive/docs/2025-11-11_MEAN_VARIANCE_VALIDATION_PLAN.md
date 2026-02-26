# Mean and Variance Prediction Validation Plan
**Date:** 2025-11-11
**Purpose:** Re-validate Nov 3 pair-level approach with proper methodology for anomaly detection

---

## Background

### The Validation Problem

Previous work (Oct 31, Nov 3, Nov 5) used **mean validation**:
- Train on mean(perms 0-N)
- Validate on mean(perms M-P)
- This inflates correlations by approximately 0.20 points

### The Anomaly Detection Requirement

For anomaly detection, we need **pair-specific z-scores**:

```
z = (observed - mu) / sigma
```

Where:
- observed = pathway count in original Hetionet for specific pair
- mu = expected count under null (predicted from degrees)
- sigma = standard deviation under null (predicted from degrees)

This requires:
1. Pair-level prediction (not bin-level due to within-bin CV=62 percent)
2. Both mean AND variance estimation
3. Validation on individual permutations (not means)

---

## Experimental Design

### Data Splits (Perms 0-20 Available)

**Training:** Perms 0 to K-1 (test K = 2, 3, 4, 5, 7, 9)
- Compute mu_train = mean(perms 0 to K-1)
- Compute sigma_train = std(perms 0 to K-1)
- Train models on degree features

**Validation:** Perms 10-14 (5 perms)
- Hyperparameter tuning only
- NOT used for variance calibration
- Ensures no data leakage

**Test:** Perms 15-20 (6 perms)
- Final evaluation on individual permutations
- Each perm evaluated separately (not averaged)
- Z-score calibration checks

### Features (5 Degree Features from Nov 3)

```python
X = [
    deg_source,
    deg_target,
    deg_source * deg_target,
    deg_source ** 2,
    deg_target ** 2
]
```

### Models

1. **model_mean:** LinearRegression or SimpleNN
   - Input: 5 degree features
   - Output: predicted mean pathway count (mu)

2. **model_std:** LinearRegression or SimpleNN
   - Input: 5 degree features
   - Output: predicted std pathway count (sigma)

---

## Training Procedure

For each K in [2, 3, 4, 5, 7, 9]:

```python
# 1. Sample 10,000 node pairs (50% with pathways, 50% random)
pairs = sample_pairs(metapath, n=10000)

# 2. Compute degree features
X = extract_degree_features(pairs)

# 3. Compute pathway counts for training perms
perm_counts = []
for perm_i in range(K):
    counts = compute_pathway_counts(pairs, perm=perm_i)
    perm_counts.append(counts)

# 4. Compute training targets
mu_train = np.mean(perm_counts, axis=0)
sigma_train = np.std(perm_counts, axis=0)

# 5. Train models
model_mean.fit(X, mu_train)
model_std.fit(X, sigma_train)
```

---

## Validation Procedure (Perms 10-14)

```python
# Used for hyperparameter tuning only
# Example: early stopping, regularization strength, etc.

for perm in range(10, 15):
    mu_pred = model_mean.predict(X)
    counts_val = compute_pathway_counts(pairs, perm=perm)
    r_val = correlation(mu_pred, counts_val)

# Select hyperparameters that maximize mean(r_val)
# Do NOT use validation perms for computing sigma or z-scores
```

---

## Test Procedure (Perms 15-20)

For each test permutation individually:

```python
results = []

for test_perm in range(15, 21):
    # Predict mean and std
    mu_pred = model_mean.predict(X)
    sigma_pred = model_std.predict(X)

    # Get actual counts for this perm
    counts_test = compute_pathway_counts(pairs, perm=test_perm)

    # Evaluate mean prediction
    r_mean = correlation(mu_pred, counts_test)
    mae_mean = mean_absolute_error(mu_pred, counts_test)

    # Compute z-scores
    z = (counts_test - mu_pred) / sigma_pred

    # Z-score calibration metrics
    z_mean = np.mean(np.abs(z))        # Should be ~0.8
    z_std = np.std(z)                  # Should be ~1.0
    z_outliers = np.mean(np.abs(z) > 3)  # Should be ~0.003

    # QQ plot correlation (normality test)
    qq_corr = stats.probplot(z)[1][2]  # Should be >0.95

    results.append({
        'K': K,
        'test_perm': test_perm,
        'r_mean': r_mean,
        'mae_mean': mae_mean,
        'z_mean': z_mean,
        'z_std': z_std,
        'z_outliers': z_outliers,
        'qq_corr': qq_corr
    })
```

---

## Success Criteria

### Mean Prediction Quality

- **r_mean > 0.80** on individual test perms
- Consistent across all 6 test perms (std < 0.05)
- MAE reasonable (< 2 pathways for typical metapaths)

### Z-Score Calibration

If z-scores are well-calibrated (i.e., null distribution prediction is accurate):

- **mean(abs(z))** in [0.7, 0.9] (theoretical value: 0.798 for N(0,1))
- **std(z)** in [0.9, 1.1] (theoretical value: 1.0 for N(0,1))
- **z_outliers** in [0.001, 0.01] (theoretical value: 0.0027 for N(0,1))
- **qq_corr > 0.95** (z-scores approximately normal)

### Minimum K Determination

Find minimum K where:
1. r_mean plateaus (change < 0.02 when adding more perms)
2. Z-score calibration is acceptable
3. Variance across test perms is low (stable predictions)

---

## Expected Outcomes

### Based on Statistical Theory

**Mean estimation:**
- Standard error decreases as 1/sqrt(K)
- K=3-5 likely sufficient for r>0.80

**Variance estimation:**
- Relative error in std decreases as 1/sqrt(2K-2)
- K=5-7 likely needed for stable sigma estimates
- Z-score calibration requires accurate variance

**Expected minimum K:** 5-7 permutations

### Computational Cost

If K=7 is minimum:
- 7 perms per metapath vs 200 in original approach
- 96.5 percent reduction in computational cost
- ~35 minutes per metapath vs 17 hours

---

## Implementation

### Script: test_src/validate_mean_variance_prediction.py

**Key functions:**
- `sample_pairs(metapath, n)` - Sample node pairs
- `extract_degree_features(pairs)` - Extract 5 degree features
- `compute_pathway_counts(pairs, perm)` - Count pathways in permutation
- `train_models(X, mu_train, sigma_train)` - Train mean and std models
- `evaluate_on_test_perms(models, X, test_perms)` - Test evaluation
- `plot_K_comparison(results)` - Visualization

### Test Metapath

**CbGpPW (Compound-Gene-Pathway):**
- Dense metapath (71,653 pairs with pathways)
- Nov 3 showed r>0.98 for this metapath (mean validation)
- Good baseline for testing methodology

---

## Outputs

### Results Files

**results/mean_variance_validation/K_comparison.csv**
- Columns: K, test_perm, r_mean, mae_mean, z_mean, z_std, z_outliers, qq_corr
- 54 rows (6 test perms × 9 K values)

**results/mean_variance_validation/K_comparison_summary.csv**
- Aggregated statistics per K
- Columns: K, mean_r, std_r, mean_z_mean, mean_z_std, mean_outliers

### Visualizations

**results/mean_variance_validation/K_vs_performance.png**
- Panel 1: K vs r_mean (with std bands)
- Panel 2: K vs z_mean (target line at 0.8)
- Panel 3: K vs z_std (target line at 1.0)
- Panel 4: K vs z_outliers (target line at 0.003)

---

## Next Steps After Validation

### If Successful (r>0.80, good calibration)

1. Determine minimum K from results
2. Scale to other metapaths (CrCbG, CtDaG, etc.)
3. Create production pipeline for anomaly detection
4. Document final methodology

### If Unsuccessful (r<0.75 or poor calibration)

1. Investigate why pair-level prediction struggles
2. Consider alternative features (add intermediate degree stats?)
3. Assess computational cost of enumeration approach
4. Re-evaluate anomaly detection strategy

---

## Data Leakage Prevention

**Critical checks:**
- Training uses ONLY perms 0 to K-1
- Validation uses ONLY perms 10-14 (hyperparameters only)
- Test uses perms 15-20 (never seen during training/validation)
- Sigma computed from training perms only (NOT validation or test)
- Z-scores evaluated on test perms using predictions (NOT empirical values)

**Each stage must be strictly separated** to avoid inflating performance estimates.

---

## Timeline

- Implementation: 30 minutes
- Execution (CbGpPW): 5 minutes
- Analysis: 15 minutes
- Documentation: 10 minutes
- **Total: ~1 hour**

---

## References

- Oct 31 RESULTS.md (bin-level approach)
- Nov 3 SESSION_SUMMARY (pair-level approach with mean validation)
- Nov 11 RESULTS.md (discovered mean validation inflation)
- GROUND_TRUTH_SUMMARY_2025-11-11.md (comprehensive audit)
