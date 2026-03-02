"""
Pair-Level Phase 2: Degree-Aware Correction

Goal: Apply degree-aware correction (successful at bin-level r > 0.99) to
      pair-level predictions to push r from 0.88-0.91 → 0.92-0.95+

Implementation
--------------
Two-stage model (adapted from bin-level Phase 5b):
1. Stage 1: Base model predicts permutation average from original graph features
2. Stage 2: Correction model learns (original, perm 0) differences
3. Validation: Test on permutations 1-20

Metapaths: CbGpPW, CtDaG, CrCbG (from Phase 1)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup paths
repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'pair_level_phase2_correction'
results_dir.mkdir(parents=True, exist_ok=True)

from pair_level_features import extract_pair_features
from pair_level_sampling import sample_pairs_stratified, compute_pair_pathway_counts

print("="*80)
print("PAIR-LEVEL PHASE 2: DEGREE-AWARE CORRECTION")
print("="*80)

# Configuration
METAPATHS = [
    ('CbG', 'GpPW', 'CbGpPW'),
    ('CtD', 'DaG', 'CtDaG'),
    ('CrC', 'CbG', 'CrCbG')
]

N_SAMPLES = 50000  # Per metapath (from Phase 1)
PERM_0 = 0  # For correction training
PERM_VALIDATION = list(range(1, 21))  # For validation
RANDOM_STATE = 42

print(f"\nConfiguration:")
print(f"  Metapaths: {[mp[2] for mp in METAPATHS]}")
print(f"  Samples per metapath: {N_SAMPLES:,}")
print(f"  Correction source: Permutation {PERM_0}")
print(f"  Validation: Permutations {PERM_VALIDATION[0]}-{PERM_VALIDATION[-1]}")


def extract_correction_features(X, y_pred):
    """
    Extract features for correction model.

    Features (15 total):
    - Degree features (7): source, target, product, source², target², sqrt(source), sqrt(target)
    - Prediction features (3): y_pred, y_pred², log(y_pred+1)
    - Interaction terms (5): y_pred × source, y_pred × target, etc.

    Parameters
    ----------
    X : ndarray (n_pairs, 5)
        Original features: [source_deg, target_deg, product, source², target²]
    y_pred : ndarray (n_pairs,)
        Base model predictions

    Returns
    -------
    ndarray (n_pairs, 15)
        Correction features
    """
    source_deg = X[:, 0]
    target_deg = X[:, 1]

    features = [
        source_deg,
        target_deg,
        source_deg * target_deg,
        source_deg ** 2,
        target_deg ** 2,
        np.sqrt(source_deg + 1),
        np.sqrt(target_deg + 1),
        y_pred,
        y_pred ** 2,
        np.log1p(y_pred)
    ]

    # Interaction terms (critical for heteroscedasticity)
    features.extend([
        y_pred * source_deg,
        y_pred * target_deg,
        y_pred * source_deg * target_deg,
        np.sqrt(y_pred + 1) * source_deg,
        np.sqrt(y_pred + 1) * target_deg
    ])

    return np.column_stack(features)


def train_corrected_model(X, y_original, y_perm0, y_validation):
    """
    Train two-stage degree-aware correction model.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n_pairs, 5)
    y_original : ndarray
        Pathway counts from original graph
    y_perm0 : ndarray
        Pathway counts from permutation 0
    y_validation : ndarray
        Pathway counts from permutations 1-20 (target)

    Returns
    -------
    dict with models and metrics
    """
    print("    Training two-stage model...")

    # Stage 1: Base model (original → permutation average)
    print("      Stage 1: Base model")
    base_model = LinearRegression()
    base_model.fit(X, y_validation)  # Train directly on validation target
    y_pred_base = base_model.predict(X)

    # Baseline metrics
    r_base = pearsonr(y_pred_base, y_validation)[0]
    rmse_base = np.sqrt(np.mean((y_pred_base - y_validation)**2))
    bias_base = np.mean(y_pred_base - y_validation)

    print(f"        Baseline: r = {r_base:.4f}, RMSE = {rmse_base:.4f}, bias = {bias_base:+.4f}")

    # Stage 2: Correction model (using permutation 0)
    print("      Stage 2: Correction model")
    correction_features = extract_correction_features(X, y_pred_base)
    correction_target = y_perm0 - y_pred_base

    correction_model = LinearRegression()
    correction_model.fit(correction_features, correction_target)
    correction = correction_model.predict(correction_features)

    y_pred_corrected = y_pred_base + correction

    # Corrected metrics
    r_corrected = pearsonr(y_pred_corrected, y_validation)[0]
    rmse_corrected = np.sqrt(np.mean((y_pred_corrected - y_validation)**2))
    bias_corrected = np.mean(y_pred_corrected - y_validation)

    print(f"        Corrected: r = {r_corrected:.4f}, RMSE = {rmse_corrected:.4f}, bias = {bias_corrected:+.4f}")
    print(f"        Improvement: Δr = {r_corrected - r_base:+.4f}, ΔRMSE = {rmse_corrected - rmse_base:+.4f}")

    return {
        'base_model': base_model,
        'correction_model': correction_model,
        'y_pred_base': y_pred_base,
        'y_pred_corrected': y_pred_corrected,
        'r_base': r_base,
        'r_corrected': r_corrected,
        'rmse_base': rmse_base,
        'rmse_corrected': rmse_corrected,
        'bias_base': bias_base,
        'bias_corrected': bias_corrected
    }


# Process each metapath
all_results = []

for edge1_type, edge2_type, metapath_name in METAPATHS:
    print(f"\n{'='*80}")
    print(f"METAPATH: {metapath_name} ({edge1_type} → {edge2_type})")
    print(f"{'='*80}")

    # Load Phase 1 data (or regenerate)
    phase1_results_file = repo_dir / 'results' / 'pair_level_models' / f'phase1_{metapath_name}_model.pkl'

    if phase1_results_file.exists():
        print(f"  Loading Phase 1 results from: {phase1_results_file.name}")
        with open(phase1_results_file, 'rb') as f:
            phase1_data = pickle.load(f)

        # Extract data from Phase 1
        X = phase1_data.get('X_train')  # May need to regenerate
        y_validation = phase1_data.get('y_validation')

        if X is None or y_validation is None:
            print("    Phase 1 data incomplete, regenerating...")
            # TODO: Regenerate from Phase 1 script
            print("    ERROR: Need to regenerate Phase 1 data")
            continue
    else:
        print("    Phase 1 results not found, need to run Phase 1 first")
        print(f"    Expected: {phase1_results_file}")
        continue

    # Load permutation 0 pathway counts
    print(f"  Loading permutation 0 pathway counts...")
    # TODO: Need to compute this from permutation 0
    # For now, skip if not available
    print("    ERROR: Need to compute permutation 0 pathway counts")
    continue

    # Train corrected model
    # result = train_corrected_model(X, y_original, y_perm0, y_validation)
    # all_results.append({
    #     'metapath': metapath_name,
    #     **result
    # })

print("\n" + "="*80)
print("PHASE 2 INCOMPLETE - MISSING DATA")
print("="*80)
print("""
Issue: Need to regenerate or load Phase 1 data with:
1. X (features)
2. y_validation (target)
3. y_perm0 (for correction)
4. pair_indices (for tracking)

Next steps:
1. Modify Phase 1 script to save required data
2. OR: Regenerate data in Phase 2 script
3. Then complete correction model training
""")
