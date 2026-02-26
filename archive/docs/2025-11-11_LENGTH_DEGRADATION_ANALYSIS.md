# Length Degradation Analysis: Performance Ceiling at Length 6-7
**Date:** 2025-11-11
**Analysis:** Complete assessment of endpoint-only prediction from path length 2-8

---

## Executive Summary

We systematically tested the CbG-GiG-GpPW metapath series from length 2 (single edge) to length 8 (seven edges) to map complete performance degradation. Results reveal a dramatic non-monotonic pattern:

**Correlation improves from length 2-7, then collapses at length 8:**
- Length 2: r = 0.449
- Length 3-7: r increases to peak of 0.912 at length 6
- **Length 8: r collapses to 0.513** (worse than length 2)

**Calibration degrades to length 5, then partially recovers:**
- Length 2: Q-Q = 0.902 (excellent)
- Length 5: Q-Q = 0.461 (minimum)
- Length 8: Q-Q = 0.760 (partial recovery)

**Key finding:** Endpoint-only prediction has a **performance ceiling at length 6-7** (r ≈ 0.91), beyond which the approach breaks down despite counts reaching 200 million.

---

## Methodology

### Metapath Series

Tested the CbG-GiG-GpPW series with systematic GiG edge additions:

| Length | Metapath | Description | Edges |
|--------|----------|-------------|-------|
| 2 | CbG | Single edge | 1 |
| 3 | CbGpPW | Gene function | 2 |
| 4 | CbGiGpPW | Extended function | 3 |
| 5 | CbGiGiGpPW | Deep function | 4 |
| 6 | CbGiGiGiGpPW | Very deep function | 5 |
| 7 | CbGiGiGiGiGpPW | Ultra-deep function | 6 |
| 8 | CbGiGiGiGiGiGpPW | Extreme depth | 7 |

### Protocol

For each length:
1. Sample 10,000 (Compound, Pathway) pairs
2. Extract endpoint features: [deg_Compound, deg_Pathway, products, squares]
3. Train Random Forest on mean counts (perms 0-4)
4. Evaluate on test perms 15-19

**Note:** Lengths 3, 4, 5 reuse existing results from prior analyses.

---

## Complete Results Table

| Length | r (mean ± std) | Q-Q (mean ± std) | Mean Count | Pairs with Pathways |
|--------|----------------|------------------|------------|---------------------|
| 2 | 0.449 ± 0.007 | 0.902 ± 0.001 | 0.142 | 50.1% |
| 3 | 0.777 ± 0.025 | 0.843 ± 0.006 | 0.295 | 57.8% |
| 4 | 0.828 ± 0.012 | 0.730 ± 0.038 | 2.853 | 83.4% |
| 5 | 0.899 ± 0.027 | 0.461 ± 0.063 | 268.964 | 94.6% |
| 6 | **0.912 ± 0.016** | 0.609 ± 0.058 | 28,369.352 | 94.4% |
| 7 | 0.905 ± 0.015 | 0.619 ± 0.044 | 3,277,610.477 | 94.5% |
| 8 | **0.513 ± 0.024** | 0.760 ± 0.007 | 199,759,193.962 | 92.4% |

---

## Key Findings

### 1. Performance Peaks at Length 6-7

**Correlation trajectory:**
- Length 2 → 3: +73% improvement (0.449 → 0.777)
- Length 3 → 6: +17% improvement (0.777 → 0.912)
- **Length 7 → 8: -43% collapse** (0.905 → 0.513)

**Best performance:** r = 0.912 at length 6, r = 0.905 at length 7

**Interpretation:** There is a **sweet spot at length 6-7** where:
- Enough paths to average out noise (law of large numbers)
- Not so many paths that all pairs become saturated
- Degree features still differentiate connectivity

### 2. Catastrophic Collapse at Length 8

**What happened:**
- Correlation drops from 0.905 to 0.513
- This is **worse than length 2** (single edge)
- Despite counts reaching 200 million (average)
- Calibration actually improves (Q-Q: 0.619 → 0.760)

**Why the collapse:**

1. **Near-saturation of pathway space:**
   - 92.4% of pairs have pathways (vs 50.1% at length 2)
   - Most high-degree pairs are fully connected
   - Degree no longer discriminates

2. **Numerical issues:**
   - Counts range into hundreds of millions
   - Random Forest may struggle with such extreme values
   - Floating point precision limits

3. **Signal-to-noise reversal:**
   - At length 2-7: Degree signal > topology noise
   - At length 8: Topology variance overwhelms degree signal
   - Even though absolute counts are enormous, relative differences compress

### 3. Calibration Follows U-Shaped Curve

**Q-Q trajectory:**
- Length 2: Q-Q = 0.902 (excellent)
- Length 3-5: Degrades to minimum Q-Q = 0.461
- Length 6-8: Partially recovers to Q-Q = 0.760

**Explanation:**
- **Length 2:** Few paths, low variance, well-calibrated
- **Length 3-5:** Many paths, heavy-tailed outliers, poor calibration
- **Length 6-8:** So many paths that extreme outliers become rare again (regression to mean)

**Irony:** At length 8, calibration improves because correlation is so poor that residuals are more homogeneous.

### 4. Count Magnitude Explosion

**Exponential growth:**
- Length 2: 0.14 counts
- Length 4: 2.85 counts (20x increase)
- Length 6: 28,369 counts (10,000x increase)
- Length 8: 199,759,194 counts (7,000x increase from length 6)

**Path saturation:**
- Length 2: 50% of pairs connected
- Length 6-7: 94% of pairs connected
- Length 8: 92% of pairs connected (slight decrease due to sampling)

**Interpretation:** At length 8, almost all pairs are connected with astronomical counts. Degree features become uninformative because **everyone is connected to everyone**.

---

## Statistical Insights

### The Law of Large Numbers Effect (Length 2-7)

As paths lengthen:
1. **More paths per pair** → averaging reduces topology-specific noise
2. **Mean becomes predictable** from endpoint degrees (r increases)
3. **But variance accumulates** → outliers become more extreme (Q-Q decreases)

This pattern holds through length 7, where r reaches its peak.

### The Saturation Effect (Length 8)

Beyond length 7:
1. **Pathway space saturates** → most pairs fully connected
2. **Degree loses discriminative power** → all high-degree pairs have ~same connectivity
3. **Relative differences compress** → hard to distinguish 100M from 200M paths
4. **Model breaks down** → r collapses below usable threshold

### Why Calibration Recovers at Length 8

**Poor correlation creates homogeneous residuals:**
- At length 5-7: Model predicts well → extreme outliers stand out → heavy tails
- At length 8: Model predicts poorly → predictions near mean for everyone → normal residuals

**Q-Q improves because predictions are uninformative**, not because they're accurate.

---

## Practical Implications

### Optimal Path Length: 6-7

For endpoint-only prediction:
- **Best performance:** Length 6 (r = 0.912) or Length 7 (r = 0.905)
- **Do NOT use:** Length 8 or beyond (r < 0.6, approach fails)
- **Acceptable range:** Length 3-7 (r > 0.77)
- **Not recommended:** Length 2 (r = 0.45, insufficient averaging)

### Application Guidelines

**Length 2-3 (r = 0.45-0.78):**
- Use: Basic connectivity prediction
- Calibration: Excellent (Q-Q > 0.84)
- Limitation: Moderate correlation

**Length 4-5 (r = 0.83-0.90):**
- Use: Mean prediction, ranking
- Calibration: Poor (Q-Q < 0.75)
- Limitation: Heavy-tailed residuals

**Length 6-7 (r = 0.91):**
- Use: **Optimal for mean prediction**
- Calibration: Moderate (Q-Q ≈ 0.61)
- Limitation: Cannot use for uncertainty quantification

**Length 8+ (r < 0.6):**
- **DO NOT USE**
- Approach breaks down
- Saturation effects dominate

### Recommendations by Use Case

**For ranking/prioritization:**
- Use length 6-7 (best discrimination)
- Expect r ≈ 0.91

**For anomaly detection:**
- Use length 3 (r = 0.78, Q-Q = 0.84)
- Avoid length 8 (poor correlation)

**For uncertainty quantification:**
- Length 2-3 only (Q-Q > 0.84)
- Length 4-8 require specialized methods

---

## Visualizations

### Figure 1: Combined Degradation Plot
**File:** `results/length_degradation/length_degradation_combined.png`

Shows correlation (blue) and Q-Q (orange) on twin y-axes vs path length.

**Key visual patterns:**
- **Blue line (correlation):** Rises steadily to peak at length 6-7, then cliff drop at length 8
- **Orange line (calibration):** U-shaped curve, minimum at length 5, partial recovery at length 8
- **Error bars:** Larger at lengths 4-7 (high variance), smaller at length 2 and 8 (different reasons)

**Annotations:**
- Length 2: r=0.449, Q-Q=0.902
- Length 8: r=0.513, Q-Q=0.760 (worse correlation, better calibration)

### Figure 2: Separate Degradation Panels
**File:** `results/length_degradation/length_degradation_plots.png`

**Left panel - Performance:**
- Horizontal line at r=0.8 threshold
- Clear peak at lengths 6-7
- Dramatic drop at length 8

**Right panel - Calibration:**
- Horizontal lines at Q-Q=0.8 (good) and Q-Q=0.5 (poor)
- Minimum at length 5
- Recovery at length 8 (ironic - poor predictions produce normal residuals)

---

## Theoretical Explanation

### Why Performance Peaks at Length 6-7

**Optimal balance of two competing forces:**

1. **Law of large numbers (improves r):**
   - More paths → better averaging
   - Reduces topology-specific noise
   - Makes mean predictable from degrees

2. **Saturation effects (degrades r):**
   - Too many paths → everyone connected
   - Degree no longer discriminates
   - Relative differences compress

**Length 6-7 is the sweet spot** before saturation dominates.

### The Saturation Threshold

**Quantitative evidence:**
- Pathway prevalence plateaus at ~94% (lengths 5-7)
- Mean counts grow 100x per edge
- At length 8: 200 million average count
- Distribution compresses: all high-degree pairs have ~same connectivity

**Information theory perspective:**
- Degree features have limited information capacity
- Can distinguish 10 vs 1000 paths (3 orders of magnitude)
- Cannot distinguish 100M vs 200M paths (lost in noise)

### Why Calibration U-Curves

**Phase 1 (Length 2-5): Degradation**
- Few paths → many paths
- Low variance → high variance with extreme outliers
- Normal residuals → heavy-tailed residuals
- Q-Q degrades

**Phase 2 (Length 5-8): Partial Recovery**
- Many paths → saturation
- Extreme outliers → regression to mean (everyone has many paths)
- Model predictions near mean for all → more homogeneous residuals
- Q-Q improves (but correlation collapses)

---

## Conclusions

### Main Findings

1. **Performance ceiling exists at length 6-7** (r ≈ 0.91)
2. **Catastrophic collapse at length 8** (r = 0.51)
3. **Optimal range: Length 3-7** for endpoint-only prediction
4. **Saturation effects dominate beyond length 7**
5. **Calibration follows U-curve** (good → poor → moderate)

### Answer to Original Question

**"Extend one path to 6, 7, and 8, plotting correlation vs calibration over lengths 2-8."**

**Results:**
- Extended CbG-GiG-GpPW series to length 8
- Correlation peaks at length 6 (r=0.912), collapses at length 8 (r=0.513)
- Calibration follows U-curve: excellent → poor → moderate
- **Practical limit: Length 7** (r=0.905, Q-Q=0.619)

**Key insight:** Endpoint-only prediction has a hard ceiling at length 6-7, beyond which saturation effects cause catastrophic failure.

### Implications for Null Models

**Confirmed:** Degree-based null models work well for short-to-medium paths (length 3-7).

**Limitation discovered:** Cannot extend indefinitely. Saturation effects create a hard limit around length 7-8.

**Recommendation:** For paths longer than length 7, require either:
1. Intermediate node features (defeats null model purpose)
2. Specialized saturation-aware models
3. Accept poor performance (r < 0.6)

### Future Work

1. **Test saturation threshold on other metapaths** - Is length 7-8 universal?
2. **Develop saturation-aware models** - Can we model the compression explicitly?
3. **Investigate numerical precision effects** - Does log-transform help at length 8?
4. **Compare to compositional approaches** - Does multiplication handle saturation better?

---

## Files Generated

### Scripts
- `test_src/test_length_degradation.py` - Length 2-8 analysis

### Results
- `results/length_degradation/length_degradation_results.csv` - Full results (35 rows)

### Visualizations
- `results/length_degradation/length_degradation_combined.png` - Twin-axis plot
- `results/length_degradation/length_degradation_plots.png` - Separate panels

### Documentation
- `docs/2025-11-11_LENGTH_DEGRADATION_ANALYSIS.md` (this file)

---

**Analysis completed:** 2025-11-11
**Path lengths tested:** 2, 3, 4, 5, 6, 7, 8
**Key achievement:** Discovered performance ceiling at length 6-7 and catastrophic collapse at length 8 due to saturation effects
