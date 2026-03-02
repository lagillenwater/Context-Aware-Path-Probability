"""
Investigate Experiment 2A Feature Values

Goal: Understand why Experiment 2A failed with r=0.62 by examining actual
feature values for diverse node pairs.

Key questions:
1. Why does naive_composition (coefficient=0.002) have no predictive power?
2. Why does max_CbGiG (coefficient=0.44) dominate?
3. Which pairs are predicted well vs poorly?

Date: 2025-11-04
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'

print("="*80)
print("INVESTIGATING EXPERIMENT 2A FEATURE VALUES")
print("="*80)


def load_edge_matrix(edge_abbrev):
    """Load edge matrix."""
    edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'
    if edge_file.exists():
        return sp.load_npz(str(edge_file))
    raise FileNotFoundError(f"Could not find edge file for {edge_abbrev}")


def sample_node_pairs(path_count_matrix, n_samples=5000, target_ratio=0.5,
                      random_state=42):
    """Sample node pairs for training/testing."""
    np.random.seed(random_state)
    n_source, n_target = path_count_matrix.shape

    nonzero_sources, nonzero_targets = path_count_matrix.nonzero()
    n_nonzero = len(nonzero_sources)

    if n_nonzero == 0:
        raise ValueError("No non-zero paths found")

    n_nonzero_sample = min(int(n_samples * target_ratio), n_nonzero)
    nonzero_idx = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
    sampled_sources_nz = nonzero_sources[nonzero_idx]
    sampled_targets_nz = nonzero_targets[nonzero_idx]

    n_random_sample = n_samples - n_nonzero_sample
    random_sources = np.random.randint(0, n_source, n_random_sample)
    random_targets = np.random.randint(0, n_target, n_random_sample)

    all_sources = np.concatenate([sampled_sources_nz, random_sources])
    all_targets = np.concatenate([sampled_targets_nz, random_targets])

    path_count_lil = path_count_matrix.tolil()
    path_counts = np.array([
        path_count_lil[s, t] for s, t in zip(all_sources, all_targets)
    ], dtype=float).flatten()

    return all_sources, all_targets, path_counts


print("\nLoading matrices...")
CbG = load_edge_matrix('CbG')
GiG = load_edge_matrix('GiG')
GpPW = load_edge_matrix('GpPW')

print("\nComputing subpath counts...")
CbGiG = CbG @ GiG
GiGpPW = GiG @ GpPW
CbGiGpPW = CbGiG @ GpPW

print(f"  CbGiG: {CbGiG.nnz:,} non-zero")
print(f"  GiGpPW: {GiGpPW.nnz:,} non-zero")
print(f"  CbGiGpPW: {CbGiGpPW.nnz:,} non-zero")

print("\nSampling pairs...")
sources_train, targets_train, y_train = sample_node_pairs(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=42
)
sources_test, targets_test, y_test = sample_node_pairs(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=123
)

print("\nBuilding features...")
CbGiG_csr = CbGiG.tocsr()
GiGpPW_csr = GiGpPW.tocsr()

feature_names = [
    'total_CbGiG', 'total_GiGpPW', 'max_CbGiG', 'max_GiGpPW',
    'n_nonzero_CbGiG', 'n_nonzero_GiGpPW', 'deg_C', 'deg_PW',
    'total_CbGiG×total_GiGpPW', 'deg_C×deg_PW', 'naive_composition',
    'deg_C²', 'deg_PW²'
]


def extract_features(src, tgt, CbG, GiG, GpPW, CbGiG_csr, GiGpPW_csr):
    """Extract all 13 features for a pair."""
    CbGiG_row = CbGiG_csr.getrow(src).toarray().flatten()
    GiGpPW_col = GiGpPW_csr.getcol(tgt).toarray().flatten()

    total_CbGiG = CbGiG_row.sum()
    total_GiGpPW = GiGpPW_col.sum()
    max_CbGiG = CbGiG_row.max()
    max_GiGpPW = GiGpPW_col.max()
    n_nonzero_CbGiG = np.count_nonzero(CbGiG_row)
    n_nonzero_GiGpPW = np.count_nonzero(GiGpPW_col)

    deg_C = CbG.getrow(src).nnz
    deg_PW = GpPW.getcol(tgt).nnz

    interaction = total_CbGiG * total_GiGpPW
    deg_product = deg_C * deg_PW
    naive_composition = np.sum(CbGiG_row * GiGpPW_col)

    return [
        total_CbGiG, total_GiGpPW, max_CbGiG, max_GiGpPW,
        n_nonzero_CbGiG, n_nonzero_GiGpPW, deg_C, deg_PW,
        interaction, deg_product, naive_composition,
        deg_C ** 2, deg_PW ** 2
    ]


X_train_list = []
for src, tgt in zip(sources_train, targets_train):
    features = extract_features(src, tgt, CbG, GiG, GpPW, CbGiG_csr, GiGpPW_csr)
    X_train_list.append(features)
X_train = np.array(X_train_list)

X_test_list = []
for src, tgt in zip(sources_test, targets_test):
    features = extract_features(src, tgt, CbG, GiG, GpPW, CbGiG_csr, GiGpPW_csr)
    X_test_list.append(features)
X_test = np.array(X_test_list)

print("\nTraining model...")
model = LinearRegression()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

print("\n" + "="*80)
print("ANALYZING DIVERSE EXAMPLES")
print("="*80)

# Select diverse pairs
CbGiGpPW_lil = CbGiGpPW.tolil()
residuals = y_test - y_test_pred

# Categories
high_count_idx = np.where(y_test > 100)[0]
medium_count_idx = np.where((y_test > 10) & (y_test <= 100))[0]
low_count_idx = np.where((y_test > 0) & (y_test <= 10))[0]
zero_count_idx = np.where(y_test == 0)[0]

# Sample from each
np.random.seed(42)
n_per_category = 5

examples = []

for category, idx_pool, name in [
    ('high', high_count_idx, 'High count (>100)'),
    ('medium', medium_count_idx, 'Medium count (10-100)'),
    ('low', low_count_idx, 'Low count (1-10)'),
    ('zero', zero_count_idx, 'Zero count')
]:
    if len(idx_pool) == 0:
        continue

    sample_size = min(n_per_category, len(idx_pool))
    sampled_idx = np.random.choice(idx_pool, sample_size, replace=False)

    print(f"\n{name}:")
    print("-" * 80)

    for i in sampled_idx:
        src = sources_test[i]
        tgt = targets_test[i]
        true_count = y_test[i]
        pred_count = y_test_pred[i]
        residual = residuals[i]

        features = X_test[i]

        print(f"\nPair: Compound {src} -> Pathway {tgt}")
        print(f"  True count: {true_count:.1f}")
        print(f"  Predicted: {pred_count:.1f}")
        print(f"  Residual: {residual:.1f}")
        print(f"  Features:")
        for fname, fval in zip(feature_names, features):
            print(f"    {fname:30s}: {fval:12.2f}")

        # Store for CSV
        examples.append({
            'category': category,
            'compound_id': src,
            'pathway_id': tgt,
            'true_count': true_count,
            'predicted_count': pred_count,
            'residual': residual,
            **{fname: fval for fname, fval in zip(feature_names, features)}
        })

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Analyze feature correlations with true counts
correlations = []
for i, fname in enumerate(feature_names):
    r = np.corrcoef(X_test[:, i], y_test)[0, 1]
    correlations.append((fname, r, model.coef_[i]))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nFeature correlations with true counts:")
for fname, r, coef in correlations[:5]:
    print(f"  {fname:30s}: r={r:7.3f}, coef={coef:10.4f}")

print("\n" + "-"*80)
print("Why does naive_composition fail?")
naive_idx = feature_names.index('naive_composition')
naive_vals = X_test[:, naive_idx]
print(f"  naive_composition correlation with true count: r={np.corrcoef(naive_vals, y_test)[0,1]:.3f}")
print(f"  naive_composition coefficient: {model.coef_[naive_idx]:.6f}")
print(f"  Range: [{naive_vals.min():.1f}, {naive_vals.max():.1f}]")
print(f"  Mean: {naive_vals.mean():.1f}")

print("\nWhy does max_CbGiG dominate?")
max_idx = feature_names.index('max_CbGiG')
max_vals = X_test[:, max_idx]
print(f"  max_CbGiG correlation with true count: r={np.corrcoef(max_vals, y_test)[0,1]:.3f}")
print(f"  max_CbGiG coefficient: {model.coef_[max_idx]:.6f}")
print(f"  Range: [{max_vals.min():.1f}, {max_vals.max():.1f}]")
print(f"  Mean: {max_vals.mean():.1f}")

print("\nComparison:")
total_idx = feature_names.index('total_CbGiG')
total_vals = X_test[:, total_idx]
print(f"  total_CbGiG correlation: r={np.corrcoef(total_vals, y_test)[0,1]:.3f}")
print(f"  max_CbGiG correlation:   r={np.corrcoef(max_vals, y_test)[0,1]:.3f}")
print(f"  naive_composition correlation: r={np.corrcoef(naive_vals, y_test)[0,1]:.3f}")

# Save results
df_examples = pd.DataFrame(examples)
df_examples.to_csv(results_dir / 'exp2a_feature_investigation.csv', index=False)

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print(f"\nSaved: {results_dir / 'exp2a_feature_investigation.csv'}")
print(f"Analyzed {len(examples)} example pairs")
