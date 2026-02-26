# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository develops algorithms for **context-aware path probability estimation** in biomedical knowledge graphs (BKGs). The work builds on the [Hetionet project](https://het.io/) and aims to:
- Approximate edge probability priors based on node degree and graph topology
- Assess the multiplicative assumption of edges in paths (compositional null hypothesis)
- Compare analytical formulas, machine learning models, and hybrid approaches

The project analyzes Hetionet v1.0, a heterogeneous biological network with 11 node types (Gene, Disease, Compound, etc.) and 24 relationship types (e.g., Compound-treats-Disease, Gene-associates-Disease).

## Environment Setup

**IMPORTANT: All Python code in this repository must be run in the `CAPP` conda environment.**

**Conda environment name**: `CAPP`

```bash
# Create environment
cd environments
conda env create -f environment.yml
# Or use the script:
bash create_env.sh

# Activate
conda activate CAPP
```

Key dependencies: Python 3.10, PyTorch, scikit-learn, hetnetpy, hetmatpy, papermill for notebook execution.

**Running Python scripts/tests**: Always use the conda environment:
```bash
# Method 1: Activate environment first
conda activate CAPP
python script.py

# Method 2: Use conda run
conda run -n CAPP python script.py

# Method 3: Use the specific conda path
/opt/miniconda3/bin/conda run -n CAPP python script.py
```

## Development Workflow

### Local Development (Jupyter Notebooks)

The primary development workflow uses numbered Jupyter notebooks in `notebooks/`:

```bash
jupyter lab
# Then navigate to notebooks/ and run sequentially
```

**Core pipeline notebooks:**
1. **0_create-hetmat.ipynb** - Downloads Hetionet v1.0 and converts to hetmat format (matrix representation)
2. **1_generate-permutations.ipynb** - Generates degree-preserving permutations of the network
3. **2_download_null_graphs.ipynb** - Downloads pre-computed permuted graphs
4. **3_edge_frequency_by_degree.ipynb** - Analyzes empirical edge probabilities stratified by node degree
5. **04_model_testing.ipynb** - Tests edge probability prediction models (Neural Network, Random Forest, Logistic Regression, Polynomial Regression)
6. **05_model_testing_summary.ipynb** - Cross-edge-type comparison of model performance
7. **06_minimum_permutations_analysis.ipynb** - Determines minimum permutations needed for ML models to converge
8. **07_minimum_permutations_summary.ipynb** - Cross-edge-type summary of minimum permutations analysis
9. **8_learned_analytical_formula*.ipynb** - Learns parameterized analytical formulas
10. **13_null_model_training.ipynb** - Trains null models on permutations 1-20
11. **14_fast_compositional_null*.ipynb** - Validates compositional null hypothesis (path probability = product of edge probabilities)
12. **15_metapath_null_distributions.ipynb** - Generates null distributions for specific metapaths
13. **16_dynamic_programming_dwpc.ipynb** - Dynamic programming implementation for DWPC calculation
14. **17_compositional_validation.ipynb** - Tests compositional calculation accuracy (FAILED - r=0.35)
15. **17b_compositional_failure_analysis.ipynb** - Analyzes why compositional approach fails

### HPC Batch Processing (SLURM Scripts)

Shell scripts in `scripts/` run notebooks via `papermill` on HPC clusters. See `scripts/README.md` for comprehensive HPC documentation.

**Key scripts:**
```bash
# Run locally or submit to SLURM
bash scripts/0_create_hetmat.sh
bash scripts/1_create_permutations.sh
bash scripts/04_model_comparison_analysis.sh  # Job array for 24 edge types
sbatch scripts/06_minimum_permutations_analysis.sh  # Job array for 24 edge types
bash scripts/07_minimum_permutations_summary.sh

# HPC pipeline submission
bash scripts/submit_all.sh  # Submits jobs with dependencies
```

**Script naming convention:**
- `0_`, `1_`, `2_`, `3_`: Core pipeline steps (single-digit numbering)
- `04_`, `05_`, `06_`, `07_`: Analysis notebooks (zero-padded for consistency)
- `13_`, `14_`, `15_`, `16_`, `17_`: Null models and validation
- `00X_` (e.g., `008_`): Specialized analyses
- Scripts with `_hpc` suffix: SLURM-specific versions (mostly deleted in favor of numbered scripts)

**Output directory:**
- Executed notebooks: `notebooks/executed/{notebook_name}_executed.ipynb`
- Previous: Used `notebooks/outputs/` (deprecated)

## Code Architecture

### Key Modules (`src/`)

**Model Definition & Training:**
- `model_comparison.py` - Model definitions (SimpleNN, RandomForest, Polynomial Logistic Regression) and training utilities
- `learned_analytical.py` - Learned parameterized analytical formulas for edge probability
- `model_training.py` - Training loop implementations
- `model_evaluation.py` - Model evaluation metrics and validation
- `model_visualization.py` - Plotting and visualization for model outputs

**Degree-Aware Analysis:**
- `degree_analysis.py` - Degree-stratified analysis and correlation computation
- `experiments.py` / `enhanced_experiments.py` - Experimental frameworks for testing models

**Data Processing:**
- `data_processing.py` / `data_processing_helpers.py` - Loading and preprocessing hetmat data
- `sampling.py` / `unique_sampling.py` - Sampling strategies for training/testing
- `download_utils.py` - Utilities for downloading permuted graphs

**Visualization:**
- `visualization.py` - General plotting utilities
- `scatter_plot_helpers.py` - Scatter plot generation for empirical vs. predicted probabilities

**Other:**
- `models.py` - Simple model interfaces
- `optimized_model.py` - Optimized implementations
- `run_edge_prediction.py` - Standalone edge prediction script
- `training.py` / `validation_utils.py` - Training and validation helpers

### Data Structure

```
data/
  ├── metagraph.json              # Hetionet schema
  ├── nodes/                      # Node lists by type
  ├── edges/                      # Edge matrices (sparse .npz format)
  └── permutations/               # Degree-preserving permuted graphs (200 permutations)
      └── {edge_type}_perm_{N}.npz

results/
  ├── null_models/                # Trained null models (.pkl)
  ├── compositional_null/         # Compositional null validation results
  ├── metapath_nulls/             # Metapath null distributions
  └── model_comparison_summary/   # Cross-edge-type comparison plots

notebooks/executed/               # Papermill output notebooks
logs/                            # HPC job logs (.out, .err)
```

### Key Concepts

**Edge Probability Models:**
- **Empirical**: Compute frequency of edges in permuted graphs, stratified by (source_degree, target_degree)
- **Analytical**: Closed-form formula `P(u,v) = f(deg_u, deg_v, graph_stats)`
- **Machine Learning**: Train models (NN, RF, LR) on (deg_u, deg_v) → empirical_probability

**Compositional Null Hypothesis:**
For a path u→v→w, does `P(path) ≈ P(u→v) × P(v→w)`? Tested in notebooks 14-15.

**Degree-Aware Path Calculation (DWPC):**
Reweight path contributions by edge probabilities conditioned on node degrees. Implemented in notebook 16.

**Minimum Permutations Analysis:**
Determine minimum number of permutations (N) needed for stable parameter learning. Analyzed in notebooks 6-7 and learned_analytical.py.

## Greene Lab Coding Standards

This repository follows the [Greene Lab coding standards](https://github.com/greenelab/onboarding). Key guidelines:

### Code Style

**Languages:**
- Python or R for analyses
- Python, R, or JavaScript for visualization
- JavaScript for webserver interfaces

**Python Style:**
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008)
- Maximum line length: 80-100 characters
- Use a linter (e.g., flake8, black) as part of development
- No emojis in code, comments, or commit messages

**Naming Conventions:**
- Use descriptive variable names (e.g., `source_degree` not `sd`)
- Avoid single-letter variables except for loop counters in simple contexts

**Code Organization:**
- All imports at the top of the file
- Constants at the beginning of the file
- Use proper whitespace (at least 2 spaces between inline comments and code)
- No unnecessary whitespace
- Avoid unnecessary code for aesthetics (e.g., matplotlib styling, custom color palettes)
- Focus on functional code; default styling is sufficient for scientific visualization

**Documentation:**
- Each file has a comment at the top describing its function and usage
- Each function has a docstring with:
  - Description of what it computes
  - Arguments (with types if helpful)
  - Return value(s)
- Comment complex algorithms with references to papers/notebooks

**Reproducibility:**
- Code that uses random seeds must be reproducible
- Set seed with a default value specified
- Example: `random_state=42` as a function parameter

**Error Handling:**
- API functions should catch and handle anticipated errors
- Provide precise error messages identifying the source (e.g., "lookup failed with PK=XYZ")

### Data Leakage Prevention

**CRITICAL: Avoid data leakage in machine learning experiments**

Data leakage occurs when information from the validation/test set influences model training, leading to overoptimistic performance estimates.

**Common sources of data leakage to avoid:**

1. **Training and evaluating on the same data**
   - NEVER use the same permutations for both training targets and evaluation targets
   - Example violation: Train on mean(perms 1-20), evaluate on mean(perms 1-20)
   - Correct approach: Train on mean(perms 1-10), evaluate on mean(perms 11-20)

2. **Using future information during training**
   - Do not compute training statistics (mean, variance, etc.) using validation data
   - Feature scaling/normalization must be fit on training set only

3. **Hyperparameter tuning on test set**
   - Use separate validation set for hyperparameter selection
   - Test set should only be used for final performance reporting

**Proper train/validation/test splits for this project:**

For permutation-based analyses:
```python
# CORRECT: Split permutations into non-overlapping sets
train_perms = range(1, 11)      # Permutations 1-10 for training
val_perms = range(11, 21)        # Permutations 11-20 for validation
test_perms = range(21, 31)       # Permutations 21-30 for testing (if available)

# Train target: mean of train_perms only
y_train = np.mean([compute_counts(perm) for perm in train_perms], axis=0)

# Evaluate on separate validation target
y_val = np.mean([compute_counts(perm) for perm in val_perms], axis=0)
```

**Always verify before claiming results:**
- Ask: "Is any information from the validation/test set used during training?"
- Check: Are the same permutations/samples used for both training and evaluation?
- Document: Clearly specify which data is used for train/val/test

### Version Control

**Licensing:**
- Default license: BSD-2-Clause Plus Patent License
- Always include LICENSE file in repository root

**Code Attribution:**
- Sign your code via commits attributable to your GitHub user
- Code taken from elsewhere must be properly acknowledged and license-compatible

**Repository Structure:**
- Repositories should be under the `greenelab` GitHub organization
- Code must be reviewed before merging to main/master branch

**Pull Request Process:**
- Do not commit directly to the default branch
- Create pull requests for all changes
- Each PR should focus on one functional area
- At least one lab member must approve before merging
- Each commit should contain all changes necessary for a particular fix or update

### Development Verification and Truthfulness

**Honest Progress Reporting:**
When working on fixes or implementations, distinguish clearly between:
- "I made an attempt" vs "I solved the problem"
- "I implemented a function" vs "I verified the function works end-to-end"
- "I think this should work" vs "I have confirmed this works"

**Verification Requirements:**
- Always require independent verification before claiming success
- Test that changes actually produce expected results, not just that code runs
- Verify end-to-end functionality, not just individual components
- Check that claimed fixes actually appear in final outputs/results

**Uncertainty and Limitations:**
- Be more honest about uncertainty and limitations
- Admit when you don't know rather than providing false confidence
- Acknowledge when a problem is harder than initially assessed
- Distinguish between theoretical understanding and practical implementation

**Evidence-Based Claims:**
- Provide concrete evidence that changes work (e.g., "correlation changed from X to Y")
- Show before/after comparisons when claiming improvements
- Verify that notebooks actually run with new functions/changes
- Confirm that output files contain expected improvements

### Source Code Quality

**Pride in Code:**
We expect code to be solid, well-written, tested, and documented. Even proof-of-concept code should inspire confidence.

**Failed Experiments:**
Repositories should contain failures and proof-of-concepts that didn't work. This prevents repeating the same mistakes.

**Reproducibility:**
- Maintain code that performs reproducible analyses
- Use makefiles, shell scripts, or other automation
- Include figure generation scripts in version control
- Make code publicly available before or concurrent with manuscript submission

## Testing

Test files in repository root:
- `test_notebook5_fix.py` - Tests for notebook 5 correlation fixes
- `test_updated_notebook.py` - General notebook execution tests

Run tests:
```bash
pytest test_notebook5_fix.py
python test_updated_notebook.py
```

## Running Analyses

### Single Edge Type Analysis
```bash
# Activate environment
conda activate CAPP

# Run model comparison for one edge type
papermill notebooks/04_model_testing.ipynb \
    notebooks/executed/04_model_testing_CbG_executed.ipynb \
    -p edge_file "CbG.sparse.npz" \
    -p edge_type "CbG"
```

### Batch Processing All Edge Types
```bash
# Uses job array (SLURM array indices 1-24)
sbatch scripts/04_model_comparison_analysis.sh

# Minimum permutations analysis
sbatch scripts/06_minimum_permutations_analysis.sh  # Job array
bash scripts/07_minimum_permutations_summary.sh  # Run after all jobs complete
```

### Running Full Pipeline
```bash
# Local (sequential notebooks)
bash scripts/0_create_hetmat.sh
bash scripts/1_create_permutations.sh
bash scripts/2_download_null_graphs.sh
sbatch scripts/3_edge_frequency_analysis.sh
sbatch scripts/04_model_comparison_analysis.sh  # Job array
bash scripts/05_model_testing_summary.sh
# ... continue with numbered scripts

# HPC (with dependencies)
bash scripts/submit_all.sh
```

## Important Implementation Details

### Model Input Features
All models use exactly 2 features:
- `source_degree`: Degree of source node
- `target_degree`: Degree of target node

Target: empirical edge probability from permutations.

### Edge Type Naming
Edge types use abbreviations:
- Node types: C=Compound, D=Disease, G=Gene, A=Anatomy, P=Pathway, etc.
- Relationships: b=binds, t=treats, a=associates, r=regulates, etc.
- Example: `CbG` = Compound-binds-Gene, `CtD` = Compound-treats-Disease

### Permutation Generation
Generates 200 degree-preserving permutations using XSwap algorithm. Each permutation preserves the degree sequence but randomizes edge placement.

### Papermill Execution
Notebooks are executed with `papermill` for parameterization:
```bash
papermill input.ipynb output.ipynb -p param1 value1 -p param2 value2
```

Output notebooks saved to `notebooks/executed/` or `notebooks/outputs/`.

### SLURM Job Arrays
Job arrays process multiple edge types in parallel:
```bash
#SBATCH --array=1-24
EDGE_FILE=${EDGE_FILES[$((SLURM_ARRAY_TASK_ID-1))]}
```

## Common Patterns

### Loading Hetmat Data
```python
import hetmatpy.hetmat
metagraph = hetmatpy.hetmat.MetaGraph.from_json("data/metagraph.json")
hetmat = hetmatpy.hetmat.HetMat("data/", metagraph)
edge_matrix = hetmat.get_adjacency_matrix("Compound", "Gene", "binds")
```

### Loading Permutations
```python
import scipy.sparse as sp
perm_matrix = sp.load_npz(f"data/permutations/CbG_perm_5.npz")
```

### Model Training Pattern
```python
from src.model_comparison import ModelCollection

models = ModelCollection(random_state=42)
model_dict = models.create_models(input_dim=2, edge_file_path="path/to/edge.npz")

# Train each model
for name, model in model_dict.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```

### Degree-Stratified Analysis
```python
from src.degree_analysis import calculate_degree_stratified_correlation

correlation = calculate_degree_stratified_correlation(
    empirical_probs,
    predicted_probs,
    source_degrees,
    target_degrees
)
```

## Session Documentation

**Long-Form Session Summaries:**
At the end of each work session, create comprehensive long-form summaries documenting:
- Overview of the session's focus and goals
- Key achievements with detailed explanations
- Detailed findings including quantitative results
- Scientific implications and recommended practices
- Files generated (scripts, results, documentation)
- Limitations and future work
- Summary statistics
- Clear conclusions with specific recommendations

Save summaries to `docs/YYYY-MM-DD_SESSION_SUMMARY.md`. These summaries should be thorough enough that someone can understand the work without reading code or running experiments.

**Long-Form Daily Plans:**
At the start of each work session (or when planning), create comprehensive long-form plans that explain:
- Context and motivation for each priority
- Detailed description of approaches and experiments
- Why certain approaches are expected to work or fail
- What questions each experiment answers
- Expected outcomes and decision criteria
- Clear next steps based on different outcomes

Save plans to `docs/YYYY-MM-DD_PLAN.md`. Write in narrative paragraphs, not just bullet points and tables. The plan should read like a research proposal that explains the reasoning behind each task.

## Git Workflow

Recent work has focused on:
- Fixing correlation plot issues (notebook 5 comparisons)
- Dynamic programming DWPC implementation (notebook 16)
- Optimizing compositional null models (notebook 14)
- Cleaning up HPC scripts (removing `_hpc` suffix, standardizing naming)

Untracked scripts (e.g., `scripts/008_learned_analytical.sh`) are newer versions replacing deleted `_hpc` scripts.
- Do not add labels like "OPTIMIZED" or "ENHANCED" to updated scripts. Just describe what the update is doing.
- Do not add this statement to any file: Greene Lab standards:
- No emojis
- PEP 8 compliant
- Comprehensive docstrings