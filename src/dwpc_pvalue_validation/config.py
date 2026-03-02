"""
Configuration for DWPC P-Value Validation Experiment

Contains experimental parameters, file paths, and metapath definitions.
Following Himmelstein et al. 2023 methodology exactly.
"""

from pathlib import Path
from typing import List, Tuple

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "dwpc_pvalue_validation"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
HETIONET_HETMAT_DIR = DATA_DIR  # True Hetionet as HetMat
METAGRAPH_PATH = DATA_DIR / "metagraph.json"
EDGES_DIR = DATA_DIR / "edges"
PERMUTATIONS_DIR = DATA_DIR / "permutations"

# Permutations are stored as HetMat directories (000.hetmat, 001.hetmat, etc.)
def get_permutation_hetmat_dir(perm_idx: int) -> Path:
    """Get path to permutation HetMat directory."""
    return PERMUTATIONS_DIR / f"{perm_idx:03d}.hetmat"

# Output paths
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Experimental parameters
DAMPING_EXPONENT = 0.5
NUM_PERMUTATIONS_NULL = 20
PERMUTATION_START = 1
PERMUTATION_END = 20
PERMUTATION_OBSERVED = 0

# Sampling parameters
PATHS_PER_CATEGORY = 100
DEGREE_QUANTILES = [0.0, 0.33, 0.67, 1.0]
DEGREE_CATEGORY_NAMES = ["Low", "Medium", "High"]

# Random seed for reproducibility
RANDOM_SEED = 42

# Metapaths to analyze
# Format: (abbreviation, description, length)
METAPATHS: List[Tuple[str, str, int]] = [
    # Length 3
    ("CbGpPW", "Compound-binds-Gene-participates-Pathway", 3),
    ("CtDaG", "Compound-treats-Disease-associates-Gene", 3),
    ("GiGaD", "Gene-interacts-Gene-associates-Disease", 3),
    # Length 4
    ("CbGpPWpG", "Compound-binds-Gene-participates-Pathway-participates-Gene", 4),
    ("CtDaGiG", "Compound-treats-Disease-associates-Gene-interacts-Gene", 4),
    # Length 5
    (
        "CbGpPWpGaD",
        "Compound-binds-Gene-participates-Pathway-participates-Gene-associates-Disease",
        5,
    ),
]


def get_degree_category_combinations():
    """
    Generate all 9 combinations of degree categories.

    Returns
    -------
    list of tuple
        All (source_category, target_category) combinations.
    """
    categories = DEGREE_CATEGORY_NAMES
    return [(src, tgt) for src in categories for tgt in categories]


def get_permutation_hetmat_dirs() -> List[Path]:
    """
    Get paths to all permutation HetMat directories.

    Returns
    -------
    list of Path
        Paths to permutation HetMat directories.
    """
    dirs = []
    for perm_idx in [PERMUTATION_OBSERVED] + list(range(PERMUTATION_START, PERMUTATION_END + 1)):
        perm_dir = get_permutation_hetmat_dir(perm_idx)
        if perm_dir.exists():
            dirs.append(perm_dir)
        else:
            raise FileNotFoundError(f"Permutation HetMat not found: {perm_dir}")
    return dirs
