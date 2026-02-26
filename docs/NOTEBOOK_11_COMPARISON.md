# Notebook 11 Series: Methodological Comparison

Quick reference for the notebook series demonstrating compositional probability calculation methods.

## Notebooks

| Notebook | Purpose | Status | Formula |
|----------|---------|--------|---------|
| **11.1** | OLD broken method | Deprecated (comparison only) | P = Σ (simple sum) |
| **11.2** | CORRECTED method | Validated | P = 1 - Π(1-p) |
| **11.3** | Degree-stratified analysis | Analysis | Loads 11.2 data |
| **11** | Production analysis | Production | Uses 11.2 method |

## The Bug and The Fix

### OLD Method (11.1) - BROKEN

```python
# Simple summation - CAN EXCEED 1.0
P(compound→pathway) = Σ_gene P(compound→gene) × P(gene→pathway)
```

**Problem:** Treats multiple gene pathways as additive
**Result:** Probabilities can exceed 1.0 (observed up to ~3.2)
**Why wrong:** Violates probability axiom that P must be ≤ 1.0

### CORRECTED Method (11.2) - FIXED

```python
# Probabilistic combination - ALWAYS ≤ 1.0
P(compound→pathway) = 1 - Π_gene (1 - P(compound→gene) × P(gene→pathway))
```

**Approach:** Treats multiple pathways as independent alternative routes
**Result:** All probabilities guaranteed ≤ 1.0
**Why correct:** Biologically realistic (redundancy) and mathematically valid

## Quick Start

1. **See the bug:** Run notebook 11.1 → observe probabilities > 1.0
2. **See the fix:** Run notebook 11.2 → verify all probabilities ≤ 1.0
3. **Compare:** Scatter plots show impossible values (11.1) vs valid range (11.2)
4. **Analyze:** Notebook 11.3 performs degree-stratified correlation analysis
5. **Production:** Notebook 11 uses corrected method on full permutation set

## Key Results

- **Correlation improvement:** OLD method correlations invalid due to impossible probabilities
- **Mathematical validity:** CORRECTED method satisfies probability axioms
- **Biological realism:** Multiple pathways provide redundancy, not additive probability

## Reference

Himmelstein et al. (2017) Systematic integration of biomedical knowledge prioritizes drugs for repurposing. eLife. https://doi.org/10.7554/eLife.26726

## Greene Lab Standards Applied

- No emojis
- Concise professional documentation
- PEP 8 compliant code
- Comprehensive docstrings with mathematical explanations
- Failed approaches documented for reference
