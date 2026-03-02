#!/usr/bin/env python
"""
Add comprehensive headers and cross-references to notebook series 11.x

This script adds proper documentation headers explaining each notebook's role
in the methodological comparison series.
"""

import json
from pathlib import Path


# Define headers for each notebook
NOTEBOOK_HEADERS = {
    '11_degree_conditioned_compositionality.ipynb': """# Degree-Conditioned Compositionality Analysis (PRODUCTION)

**Notebook Purpose:** Production analysis using corrected Option A probabilistic combination method.

## Overview

This notebook analyzes how the compositional assumption varies with node degrees:
1. Tests whether PMI depends systematically on compound and pathway degrees
2. Compares Hetionet (real network) with degree-preserving permutations
3. Determines if degree structure alone explains conditional dependencies

## Methodological Context

**This is part of a notebook series demonstrating methodological evolution:**
- **[11.1_empirical_vs_analytical_compositional.ipynb](11.1_empirical_vs_analytical_compositional.ipynb)**: OLD broken summation method (probabilities > 1.0)
- **[11.2_empirical_vs_analytical_compositional.ipynb](11.2_empirical_vs_analytical_compositional.ipynb)**: CORRECTED Option A method (probabilities <= 1.0)
- **[11.3_empirical_vs_analytical_compositional.ipynb](11.3_empirical_vs_analytical_compositional.ipynb)**: Degree-stratified correlation analysis using corrected data
- **This notebook (11)**: Production version with full permutation analysis

## Key Method: Option A Probabilistic Combination

The compositional probability for a metapath compound→gene→pathway is computed using:

```
P(compound→pathway) = 1 - ∏_gene (1 - P(compound→gene) × P(gene→pathway))
```

This treats multiple gene pathways as independent alternative routes with redundancy,
ensuring all probabilities remain ≤ 1.0.

**See notebook 11.1 for comparison with the broken summation approach.**

## Research Questions

1. **Is compositionality degree-dependent?** Does PMI vary systematically with node degrees?
2. **Are null (permuted) metapaths compositional?** Do degree-preserving permutations preserve or break compositionality?

## Key Findings from Previous Analysis

- **Hetionet CbGpPW**: Mean PMI = 7.11 (strongly conditional)
- **Correlation**: r = 0.057 (compositional model fails)

## This Analysis

We test:
1. Whether PMI depends on node degrees (compound degree, pathway degree)
2. Whether permuted networks show the same conditional structure
3. If degree structure alone explains the conditional dependencies
""",

    '11.1_empirical_vs_analytical_compositional.ipynb': """# Empirical vs Analytical Compositional Analysis (OLD METHOD - FOR COMPARISON)

**Notebook Purpose:** Demonstrate the mathematically invalid OLD summation method that produces probabilities > 1.0

**Status:** DEPRECATED - For educational/comparison purposes only

## Warning

This notebook intentionally implements a **broken method** to demonstrate why it fails.
Do not use this approach for production analysis.

## Overview

This notebook shows the mathematical impossibility of the original compositional formula:

```
P(compound→pathway) = Σ_gene P(compound→gene) × P(gene→pathway)
```

**Problem:** Simple summation treats multiple gene pathways as additive, which can produce
probabilities exceeding 1.0 (up to ~3.2 in our analysis).

## Expected Results

Scatter plots in Section 6.1 will show compositional probabilities exceeding 1.0,
which is mathematically impossible and violates fundamental probability axioms.

## Methodological Series Context

**This notebook is part of a methodological comparison series:**
- **This notebook (11.1)**: OLD broken summation method (demonstrates the bug)
- **[11.2_empirical_vs_analytical_compositional.ipynb](11.2_empirical_vs_analytical_compositional.ipynb)**: CORRECTED Option A method (fixes the bug)
- **[11.3_empirical_vs_analytical_compositional.ipynb](11.3_empirical_vs_analytical_compositional.ipynb)**: Analysis using corrected data
- **[11_degree_conditioned_compositionality.ipynb](11_degree_conditioned_compositionality.ipynb)**: Production version

## Why This Matters

Comparing the OLD (this notebook) vs NEW (notebook 11.2) methods demonstrates:
1. The importance of mathematical validity in probability models
2. How simple implementation errors can produce impossible results
3. The correct approach using probabilistic combination

## What Gets Fixed in Notebook 11.2

The corrected Option A formula treats multiple pathways as independent alternative routes:

```
P(compound→pathway) = 1 - ∏_gene (1 - P(compound→gene) × P(gene→pathway))
```

This ensures all probabilities remain ≤ 1.0 and is biologically realistic (multiple
pathways provide redundancy, not additive probability).

## Greene Lab Standards

This analysis follows Greene Lab coding standards:
- PEP 8 compliant
- Comprehensive docstrings
- No emojis
- Clear documentation of failed approaches for future reference
""",

    '11.2_empirical_vs_analytical_compositional.ipynb': """# Empirical vs Analytical Compositional Analysis (OPTION A - CORRECTED)

**Notebook Purpose:** Demonstrate the mathematically valid CORRECTED Option A probabilistic combination method

## Overview

This notebook implements the **corrected** Option A formula that ensures all probabilities ≤ 1.0:

```
P(compound→pathway) = 1 - ∏_gene (1 - P(compound→gene) × P(gene→pathway))
```

## Methodological Context

**This is part of a notebook series demonstrating methodological evolution:**
- **[11.1_empirical_vs_analytical_compositional.ipynb](11.1_empirical_vs_analytical_compositional.ipynb)**: OLD broken summation method (probabilities > 1.0)
- **This notebook (11.2)**: CORRECTED Option A method (probabilities <= 1.0)
- **[11.3_empirical_vs_analytical_compositional.ipynb](11.3_empirical_vs_analytical_compositional.ipynb)**: Degree-stratified correlation analysis using this corrected data
- **[11_degree_conditioned_compositionality.ipynb](11_degree_conditioned_compositionality.ipynb)**: Production version

## What Was Fixed

**Old approach (notebook 11.1):** Simple summation
```
P(compound→pathway) = Σ_gene P(compound→gene) × P(gene→pathway)
```
- Treats multiple pathways as additive
- Produces probabilities > 1.0 (up to ~3.2)
- Mathematically invalid

**New approach (this notebook):** Probabilistic combination
```
P(compound→pathway) = 1 - ∏_gene (1 - P(compound→gene) × P(gene→pathway))
```
- Treats multiple pathways as independent alternative routes
- All probabilities guaranteed ≤ 1.0
- Biologically realistic (redundancy, not additivity)
- Mathematically valid

## Expected Results

- All compositional probabilities will be ≤ 1.0 (no mathematical impossibilities)
- Results can be safely used in downstream analyses
- Correlation with empirical frequencies should be more meaningful

## Key Validation

Section 6.1 scatter plots verify that all compositional probabilities remain within [0, 1],
demonstrating the mathematical validity of Option A.

## Downstream Usage

The corrected data generated by this notebook is used in:
- Notebook 11.3 for degree-stratified correlation analysis
- Notebook 11 for production-level permutation analysis

## Greene Lab Standards

This analysis follows Greene Lab coding standards:
- PEP 8 compliant
- Comprehensive docstrings with mathematical explanations
- No emojis
- Professional, concise output messages
""",

    '11.3_empirical_vs_analytical_compositional.ipynb': """# Empirical vs Analytical Compositional Analysis (DEGREE-STRATIFIED CORRELATIONS)

**Notebook Purpose:** Analyze correlations between empirical and analytical pathway frequencies stratified by degree bins

## Overview

This notebook performs degree-stratified correlation analysis using pre-computed data
from notebook 11's degree-conditioned compositionality analysis.

## Methodological Context

**This is part of a notebook series:**
- **[11.1_empirical_vs_analytical_compositional.ipynb](11.1_empirical_vs_analytical_compositional.ipynb)**: OLD broken summation method (for comparison)
- **[11.2_empirical_vs_analytical_compositional.ipynb](11.2_empirical_vs_analytical_compositional.ipynb)**: CORRECTED Option A method
- **This notebook (11.3)**: Degree-stratified analysis using corrected data
- **[11_degree_conditioned_compositionality.ipynb](11_degree_conditioned_compositionality.ipynb)**: Production version (generates the data this notebook loads)

## Data Source

This notebook **loads pre-computed results** from notebook 11 rather than recomputing
expensive pathway calculations. Files loaded:
- `metapath_CbGpPW_hetionet_degree_results.csv`
- `metapath_CbGpPW_null_degree_results.csv`
- `metapath_CbGpPW_degree_conditioned_summary.csv`

## Analysis Approach

1. Load saved results from notebook 11's degree-conditioned analysis
2. Group pathway pairs by degree bins (same binning as notebook 11)
3. Compute correlations within each degree bin combination
4. Show complementary view of notebook 11's PMI findings using correlation analysis
5. Validate expected PMI-correlation relationship: High PMI bins → Low correlations

## Key Research Question

**Where does the independence assumption (compositionality) work vs fail?**

By stratifying by degree product and computing correlations, we reveal:
- Low-degree pairs: Compositional model performs poorly (high PMI, low correlation)
- High-degree pairs: Compositional model may work better (low PMI, higher correlation)

## Visualization Features

- Scatter plots colored by degree product show where independence assumption breaks down
- Heatmaps reveal systematic patterns in model performance across degree ranges
- PMI-correlation inverse relationship validates theoretical expectations

## Greene Lab Standards

This analysis follows Greene Lab coding standards:
- PEP 8 compliant
- Comprehensive docstrings
- No emojis
- Reproducible analysis loading from standardized CSV files
"""
}


def update_notebook_header(notebook_path, new_header):
    """
    Update the first markdown cell of a notebook with a new header.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file
    new_header : str
        New header content

    Returns
    -------
    bool
        True if updated, False if no change needed
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    if not cells:
        return False

    # Update first markdown cell
    first_cell = cells[0]
    if first_cell.get('cell_type') == 'markdown':
        first_cell['source'] = new_header

        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        return True

    return False


def main():
    """Add headers to all notebooks in the series."""
    repo_dir = Path(__file__).parent
    notebooks_dir = repo_dir / 'notebooks'

    print("="*70)
    print("ADDING COMPREHENSIVE HEADERS TO NOTEBOOK SERIES")
    print("="*70)

    for notebook_file, header_text in NOTEBOOK_HEADERS.items():
        notebook_path = notebooks_dir / notebook_file

        if notebook_path.exists():
            print(f"\nUpdating: {notebook_file}")
            updated = update_notebook_header(notebook_path, header_text)
            if updated:
                print(f"  Header updated successfully")
            else:
                print(f"  Warning: Could not update header")
        else:
            print(f"\nSkipping: {notebook_file} (not found)")

    print("\n" + "="*70)
    print("HEADERS UPDATED")
    print("="*70)
    print("All notebooks now have comprehensive headers with cross-references.")
    print("="*70)


if __name__ == '__main__':
    main()
