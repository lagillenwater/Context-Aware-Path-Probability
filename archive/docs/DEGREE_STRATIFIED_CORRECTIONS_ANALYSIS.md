# Degree-Stratified Corrections Analysis

## Motivation

Initial residual analysis showed that uniform theoretical corrections helped low/mid-degree pairs but hurt high-degree pairs in AeG:

| Degree Range | % Improved | Mean Residual Change |
|-------------|-----------|---------------------|
| Low (0-33%) | 93.6% | -0.003317 → -0.003345 (tiny worsening) |
| Mid (33-67%) | 69.6% | -0.047814 → -0.049068 (tiny worsening) |
| High (67-100%) | 8.1% | -0.189046 → -0.245137 (significant worsening) |
| Hub-hub pairs | 0.9% | -0.095 → -0.263 (much worse) |

**Goal**: Apply different correction strategies for different degree ranges to improve overall performance.

## Approaches Tested

### Approach 1: Uniform Corrections (Baseline)

**Strategy**: Apply all corrections (assortativity, heterogeneity, density) uniformly across all degree ranges.

**Results**:
- CbG: r = 0.9905 → 0.9833 (worse by 0.0071)
- AeG: r = 0.9598 → 0.9628 (BETTER by 0.0031) ✓

**Analysis**: Works best for AeG despite hurting high-degree pairs. The improvement in low/mid-degree pairs (which are more numerous) outweighs the degradation in high-degree pairs.

### Approach 2: Skip All Corrections for High-Degree Pairs

**Strategy**:
- Low/mid degree (0-67%): Apply all corrections
- High degree (67-100%): No corrections (leave analytical predictions unchanged)

**Results**:
- CbG: r = 0.9905 → 0.9898 (worse by 0.0007)
- AeG: r = 0.9598 → 0.9599 (tiny improvement +0.0001)

**Analysis**: Protects high-degree pairs from getting worse, but reduces overall benefit. The tiny improvement for AeG (+0.0001) is much worse than uniform corrections (+0.0031).

### Approach 3: Heterogeneity-Only for High-Degree Pairs

**Strategy**:
- Low/mid degree (0-67%): Apply all corrections
- High degree (67-100%): Apply only heterogeneity correction (skip assortativity and density)

**Rationale**: Heterogeneity correction increases predictions for hubs (helps with underestimation), while assortativity and density corrections decrease predictions (make underestimation worse).

**Results**:
- CbG: r = 0.9905 → 0.9841 (worse by 0.0064)
- AeG: r = 0.9598 → 0.9570 (worse by 0.0028)
  - But bias improved: -0.0797 → -0.0777 (reduction of 0.0021)

**Analysis**: Heterogeneity correction helps reduce bias but hurts correlation. Still worse than uniform corrections.

## Key Findings

### 1. Uniform Corrections Remain Best

Despite hurting high-degree pairs, uniform corrections achieve the best overall correlation improvement for AeG (+0.0031). This is because:

- Low/mid-degree pairs are more numerous
- Improvements in these ranges outweigh degradation in high-degree range
- Correlation is a global metric sensitive to the majority of points

### 2. Why High-Degree Pairs Get Worse

For high-degree pairs under negative assortativity:

**Assortativity correction**: Reduces predictions for high×high pairs (makes underestimation worse)

**Density correction**: Reduces predictions via `1 - gamma * density * log(1 + u*v)` (makes underestimation worse)

**Heterogeneity correction**: Increases predictions only for hub-hub pairs (both > 90th percentile), which is a small subset

Result: 2 out of 3 corrections make things worse, and the helpful one (heterogeneity) only applies to a small subset.

### 3. Distribution of Bias

The analytical formula has different bias patterns across degree ranges:

**CbG**:
- Low: -0.000435 (near-zero, excellent)
- Mid: -0.005292 (small underestimation)
- High: +0.003804 (small overestimation)
- Overall: Very well-predicted (r = 0.9905)

**AeG**:
- Low: -0.003317 (near-zero, excellent)
- Mid: -0.047814 (moderate underestimation)
- High: -0.189046 (large underestimation)
- Overall: Systematic underestimation (r = 0.9598)

For AeG, the high-degree underestimation is the largest problem, but corrections make it worse rather than better.

### 4. Limitations of Simple Multiplicative Corrections

The failure of degree-stratified corrections reveals limitations:

**Cannot capture complex interactions**: Assortativity and heterogeneity effects may interact non-linearly across degree ranges

**Cannot reverse systematic bias**: Simple multiplicative factors can't increase predictions for high-degree pairs without hurting low/mid-degree pairs

**Limited flexibility**: Only 3 correction factors (assortativity, heterogeneity, density) can't capture degree-dependent patterns

## Recommendations

### For Current Implementation

**Use uniform corrections with `degree_stratified=False` (default):**

```python
from src.theoretical_corrections import apply_all_corrections, extract_graph_features

graph_features = extract_graph_features(edge_matrix)
P_corrected = apply_all_corrections(
    P_analytical, u, v, graph_features, degree_stratified=False
)
```

**When to use**:
- Edge types with systematic bias (r < 0.97)
- Accept modest improvement (+0.003 for AeG)
- Accept that high-degree pairs may get worse

**When NOT to use**:
- Edge types already well-predicted (r > 0.99, like CbG)
- Corrections will hurt performance

### For Future Work

**Option 1: Physics-Informed Neural Network**

Train NN using only theoretical constraints (no empirical data):

```python
loss = w1 * conservation_loss +      # Total predicted edges = actual edge count
       w2 * marginal_loss +           # Marginal distributions match observed
       w3 * detailed_balance_loss +   # XSwap detailed balance
       w4 * monotonicity_loss         # Higher degree product → higher frequency
```

This could learn degree-dependent corrections while respecting XSwap theory.

**Option 2: Degree-Dependent Correction Parameters**

Instead of fixed parameters (alpha=0.1, beta=0.05, gamma=0.05), learn degree-dependent functions:

```python
alpha(u, v) = alpha_0 * f(u, v, graph_features)
beta(u, v) = beta_0 * g(u, v, graph_features)
gamma(u, v) = gamma_0 * h(u, v, graph_features)
```

But this requires some form of calibration, which brings back the "no empirical data" constraint problem.

**Option 3: Accept Current Performance**

Analytical formula achieves:
- CbG: r = 0.9905 (excellent, no corrections needed)
- AeG: r = 0.9628 with corrections (good, +0.0031 improvement)

For many applications, this may be sufficient. The remaining 4% error could be:
- Irreducible noise in empirical estimates
- Subtle effects not capturable with simple corrections
- Acceptable level of approximation error

## Implementation Details

The degree-stratified correction function is available in `src/theoretical_corrections.py`:

```python
def apply_degree_stratified_corrections(P_analytical, u, v, graph_features):
    """
    Apply degree-range-specific corrections.

    Currently kept for experimental purposes but not used by default.
    Use apply_all_corrections(..., degree_stratified=True) to enable.
    """
    # Split by 33rd and 67th percentiles
    # Apply different corrections to each range
    # See code for details
```

**Note**: Default is `degree_stratified=False` (uniform corrections) as this performs best.

## Conclusion

Degree-stratified corrections were unable to improve upon uniform corrections. The best approach remains:

1. **Use analytical formula without corrections** for well-predicted edge types (r > 0.99)
2. **Use uniform theoretical corrections** for problematic edge types (r < 0.97)
3. **Accept modest improvements** (+0.003 for AeG) as the limit of simple multiplicative corrections

Further improvements would require more sophisticated approaches (physics-informed NN, non-linear corrections, or empirical calibration).

## Files Generated

- `src/theoretical_corrections.py`: Implementation (contains both uniform and degree-stratified functions)
- `analyze_residual_differences.py`: Analysis script
- `results/theoretical_corrections/*/residual_analysis.png`: Visualization
- `results/theoretical_corrections/*/residual_analysis.csv`: Detailed data
- `THEORETICAL_CORRECTIONS_RESULTS.md`: Original results summary
- `DEGREE_STRATIFIED_CORRECTIONS_ANALYSIS.md`: This document
