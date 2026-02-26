"""
DWPC P-Value Validation Module

This module validates the DWPC p-value calculation methodology from
Himmelstein et al. 2023 (GigaScience) using permutation-based ground truth.

Two experimental scenarios:
- Scenario A: Permutation 0 vs permutations 1-20 (null test)
- Scenario B: True Hetionet vs permutations 1-20 (positive control)
"""

__version__ = "0.1.0"
__author__ = "Lucas Gillenwater"

from . import config
from . import data_loading
from . import sampling
from . import dwpc_calculation
from . import pvalue_calculation
from . import null_distribution
from . import calibration
from . import comparison
from . import visualization
from . import utils

__all__ = [
    "config",
    "data_loading",
    "sampling",
    "dwpc_calculation",
    "pvalue_calculation",
    "null_distribution",
    "calibration",
    "comparison",
    "visualization",
    "utils",
]
