# Assortativity Analysis Plan

**Date**: 2025-11-01
**Status**: Standalone analysis
**Goal**: Measure degree assortativity in Hetionet edges and pathways, compare original graph to permutation 000

---

## Objective

Quantify the tendency of nodes to connect with similar-degree nodes in Hetionet, and determine whether degree-preserving permutations maintain assortativity patterns.

---

## Background: Assortativity

**Assortativity** is the tendency of nodes to connect with other nodes of similar degree.

**Assortativity coefficient** (r):
- r > 0: Assortative (high-degree connects to high-degree)
- r = 0: Neutral (random mixing)
- r < 0: Disassortative (high-degree connects to low-degree)

**Computation**: Pearson correlation between degrees of connected node pairs.

---

## Analysis Components

### 1. Edge-Specific Assortativity

**Edges to analyze**:
- CbG (Compound-binds-Gene)
- GpPW (Gene-participates-Pathway)

**Method**:
- Load edge adjacency matrix
- Convert to NetworkX graph
- Compute degree_assortativity_coefficient()
- Compare original vs permutation 000

**Expected patterns**:
- Biological networks are typically disassortative (r < 0)
- Permutations may alter assortativity despite preserving degree sequence

### 2. Pathway-Specific Assortativity

**Metapath**: CbGpPW (Compound → Gene → Pathway)

**Three metrics**:

1. **Source-Intermediate assortativity**:
   - For each path C→G→P, measure correlation between deg(C) and deg(G)
   - Quantifies: Do high-degree compounds connect through high-degree genes?

2. **Intermediate-Target assortativity**:
   - For each path C→G→P, measure correlation between deg(G) and deg(P)
   - Quantifies: Do high-degree genes connect to high-degree pathways?

3. **Source-Target assortativity (pathway-weighted)**:
   - Correlation between deg(C) and deg(P), weighted by pathway counts
   - Quantifies: Do high-degree source/target pairs have more pathways?

**Method**:
- Enumerate all paths C→G→P
- For each path, record (deg_C, deg_G, deg_P)
- Compute pairwise correlations
- Weight by pathway counts for source-target correlation

---

## Research Questions

### RQ1: Edge-Level Assortativity
- Are CbG and GpPW edges assortative or disassortative?
- Does permutation preserve assortativity patterns?

### RQ2: Pathway-Level Assortativity
- Do high-degree compounds preferentially connect through high-degree genes?
- Is pathway assortativity preserved after permutation?

### RQ3: Permutation Effects
- XSwap preserves degree sequence but does it preserve assortativity?
- If assortativity changes, this could explain pair-level model performance

---

## Implementation

### Script: test_src/analyze_assortativity.py

**Inputs**:
- data/edges/CbG.sparse.npz
- data/edges/GpPW.sparse.npz
- data/permutations/000.hetmat/edges/CbG.sparse.npz
- data/permutations/000.hetmat/edges/GpPW.sparse.npz

**Outputs**:
- Terminal output with all assortativity coefficients
- Summary statistics

**Functions**:
1. `compute_edge_assortativity(edge_matrix)` → r
2. `compute_pathway_assortativity(edge1, edge2)` → (r_source_intermediate, r_intermediate_target, r_source_target_weighted)

---

## Expected Outcomes

### Edge Assortativity
- Biological networks typically show r < 0 (disassortative)
- Protein-protein: r ≈ -0.15 to -0.30
- Gene regulatory: r ≈ -0.20

### Pathway Assortativity
- If disassortative: High-degree compounds bind low-degree genes
- This creates "hub-and-spoke" structures
- Pathway counts may be more uniform than degree product predicts

### Permutation Effects
- If r_original ≠ r_permutation: Permutation alters assortativity
- This could explain why pair-level models (r = 0.88-0.91) don't achieve perfect correlation

---

## Relevance to Pair-Level Models

**Current Phase 1 models** use features:
- deg_source
- deg_target
- deg_source × deg_target
- deg_source²
- deg_target²

**These features assume** pathway count depends only on endpoint degrees.

**If assortativity is strong**, pathway count also depends on intermediate node degrees, which correlate with endpoint degrees in assortative/disassortative networks.

**Hypothesis**: Unexplained variance (r = 0.88-0.91, not 1.0) may come from assortativity effects not captured by endpoint degrees alone.

**Phase 2 correction** may implicitly learn assortativity patterns if bias correlates with assortativity.

---

## Metrics

| Metric | Graph | Expected Range | Interpretation |
|--------|-------|----------------|----------------|
| r_CbG | Original | -0.3 to 0.0 | Disassortative (typical) |
| r_CbG | Perm 000 | ? | May differ from original |
| r_GpPW | Original | -0.3 to 0.0 | Disassortative (typical) |
| r_GpPW | Perm 000 | ? | May differ from original |
| r_source_intermediate | Original | ? | Novel metric |
| r_intermediate_target | Original | ? | Novel metric |
| r_source_target_weighted | Original | ? | Novel metric |

---

## Deliverables

1. **Analysis script**: test_src/analyze_assortativity.py
2. **Results document**: docs/2025-11-01_ASSORTATIVITY_RESULTS.md
3. **Key findings**:
   - Edge assortativity coefficients (4 values)
   - Pathway assortativity coefficients (6 values)
   - Interpretation of patterns
   - Implications for pair-level modeling

---

## Timeline

- Script development: 10 minutes
- Analysis execution: 5 minutes
- Results documentation: 15 minutes
- Total: 30 minutes
