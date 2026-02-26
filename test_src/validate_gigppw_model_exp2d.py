"""
Experiment 2D: Validate GiGpPW Model

Validate that GiGpPW (Gene→Gene→Pathway) can be predicted with r > 0.95
using yesterday's methodology.

Date: 2025-11-04
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import time
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2D: VALIDATE GiGpPW MODEL")
print("Using Yesterday's Validated Methodology")
print("="*80)


def load_edge_matrix(edge_abbrev, perm_num='original'):
    """Load edge matrix for specific permutation."""
    if perm_num == 'original':
        edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'

    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    else:
        print(f"  Warning: {edge_file} not found")
        return None


def sample_pairs_stratified(pathway_matrix, n_samples=10000, random_state=42):
    """Sample node pairs stratified by pathway count."""
    np.random.seed(random_state)

    sources_nonzero, targets_nonzero = pathway_matrix.nonzero()
    n_nonzero = len(sources_nonzero)

    n_nonzero_sample = min(int(n_samples * 0.5), n_nonzero)
    if n_nonzero > 0:
        idx_nonzero = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
        sampled_sources = list(sources_nonzero[idx_nonzero])
        sampled_targets = list(targets_nonzero[idx_nonzero])
    else:
        sampled_sources = []
        sampled_targets = []

    n_random = n_samples - len(sampled_sources)
    random_sources = np.random.randint(0, pathway_matrix.shape[0], n_random)
    random_targets = np.random.randint(0, pathway_matrix.shape[1], n_random)

    sampled_sources.extend(random_sources)
    sampled_targets.extend(random_targets)

    return list(zip(sampled_sources, sampled_targets))


def extract_degree_features(source_idx, target_idx, edge1, edge2):
    """Extract 5 degree features."""
    d_u = edge1.getrow(source_idx).nnz
    d_v = edge2.getcol(target_idx).nnz

    return np.array([
        d_u, d_v, d_u * d_v, d_u ** 2, d_v ** 2
    ], dtype=np.float64)


print("\n" + "="*80)
print("PHASE 1: Sample Node Pairs and Extract Degree Features")
print("="*80)

print("\nLoading perm 0 edges...")
GiG_0 = load_edge_matrix('GiG', perm_num=0)
GpPW_0 = load_edge_matrix('GpPW', perm_num=0)

print(f"  GiG: {GiG_0.shape}, {GiG_0.nnz:,} edges")
print(f"  GpPW: {GpPW_0.shape}, {GpPW_0.nnz:,} edges")

print("\nComputing GiGpPW for perm 0...")
t0 = time.time()
GiGpPW_0 = GiG_0 @ GpPW_0
t_compute = time.time() - t0

print(f"  GiGpPW: {GiGpPW_0.shape}, {GiGpPW_0.nnz:,} non-zero")
print(f"  Computation time: {t_compute:.3f}s")

print("\nSampling node pairs...")
pairs = sample_pairs_stratified(GiGpPW_0, n_samples=10000, random_state=42)
print(f"  Sampled {len(pairs)} pairs")

print("\nExtracting degree features from perm 0...")
X_features = []
for src, tgt in pairs:
    features = extract_degree_features(src, tgt, GiG_0, GpPW_0)
    X_features.append(features)
X_features = np.array(X_features)
print(f"  Feature matrix: {X_features.shape}")


print("\n" + "="*80)
print("PHASE 2: Compute Training Target (Perm 0 Counts)")
print("="*80)

print("\nExtracting pathway counts from perm 0...")
GiGpPW_0_lil = GiGpPW_0.tolil()
y_train_target = np.array([
    GiGpPW_0_lil[src, tgt] for src, tgt in pairs
], dtype=float).flatten()

print(f"  Counts shape: {y_train_target.shape}")
print(f"  Non-zero: {np.sum(y_train_target > 0)}")
print(f"  Range: [{y_train_target.min():.1f}, {y_train_target.max():.1f}]")
print(f"  Mean: {y_train_target.mean():.3f}")


print("\n" + "="*80)
print("PHASE 3: Compute Validation Target (Mean of Perms 6-20)")
print("="*80)

print("\nComputing pathway counts for perms 6-20...")
val_perm_counts = []

for perm_num in range(6, 21):
    print(f"  Permutation {perm_num:02d}...", end=" ")

    GiG_perm = load_edge_matrix('GiG', perm_num=perm_num)
    GpPW_perm = load_edge_matrix('GpPW', perm_num=perm_num)

    if GiG_perm is None or GpPW_perm is None:
        print("SKIP (files not found)")
        continue

    GiGpPW_perm = GiG_perm @ GpPW_perm
    GiGpPW_perm_lil = GiGpPW_perm.tolil()

    counts = np.array([
        GiGpPW_perm_lil[src, tgt] for src, tgt in pairs
    ], dtype=float).flatten()

    val_perm_counts.append(counts)
    print(f"mean={counts.mean():.3f}")

y_val_target = np.mean(val_perm_counts, axis=0)
print(f"\nValidation target (mean of {len(val_perm_counts)} perms):")
print(f"  Shape: {y_val_target.shape}")
print(f"  Non-zero: {np.sum(y_val_target > 0)}")
print(f"  Range: [{y_val_target.min():.1f}, {y_val_target.max():.1f}]")
print(f"  Mean: {y_val_target.mean():.3f}")

r_targets = pearsonr(y_train_target, y_val_target)[0]
print(f"\nTarget correlation (perm 0 vs mean 6-20): r = {r_targets:.4f}")


print("\n" + "="*80)
print("PHASE 4: Train Model to Predict Validation Target")
print("="*80)

print("\nSplitting into train/test (80/20)...")
X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
    X_features, y_train_target, y_val_target, test_size=0.2, random_state=42
)

print(f"  Train: {len(X_train)} pairs")
print(f"  Test: {len(X_test)} pairs")

print("\nTraining model on perm 0 counts...")
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
r_train = pearsonr(y_train_pred, y_train)[0]
print(f"  Train r (perm 0): {r_train:.4f}")

print("\nEvaluating on held-out pairs...")
y_test_pred = model.predict(X_test)

r_test_train_target = pearsonr(y_test_pred, y_test)[0]
r_test_val_target = pearsonr(y_test_pred, y_val_test)[0]

print(f"\nTest Performance:")
print(f"  r vs perm 0 counts: {r_test_train_target:.4f}")
print(f"  r vs mean 6-20 counts: {r_test_val_target:.4f}")

if r_test_val_target > 0.95:
    status = "SUCCESS"
    print(f"  Status: SUCCESS (r > 0.95)")
elif r_test_val_target > 0.90:
    status = "ACCEPTABLE"
    print(f"  Status: ACCEPTABLE (r > 0.90)")
else:
    status = "FAILURE"
    print(f"  Status: FAILURE (r < 0.90)")


print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

print(f"\nTarget correlation: r = {r_targets:.4f}")
print(f"Model performance: r = {r_test_val_target:.4f}")

if r_test_val_target > 0.95:
    final_status = "VALIDATED"
    print(f"\nFINAL STATUS: VALIDATED")
    print(f"  GiGpPW model achieves r > 0.95")
    print(f"  Ready to proceed with hierarchical composition")
elif r_test_val_target > 0.90:
    final_status = "ACCEPTABLE"
    print(f"\nFINAL STATUS: ACCEPTABLE")
    print(f"  Model performs well (r > 0.90) but below threshold")
else:
    final_status = "FAILED"
    print(f"\nFINAL STATUS: FAILED")
    print(f"  Model does not achieve r > 0.90")


# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'experiment': 'Experiment 2D - GiGpPW',
    'description': 'GiGpPW validation with yesterday\'s methodology',
    'metapath': 'GiGpPW',
    'n_pairs': len(pairs),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'r_targets': r_targets,
    'r_train': r_train,
    'r_test_train_target': r_test_train_target,
    'r_test_val_target': r_test_val_target,
    'status': status,
    'final_status': final_status
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2d_gigppw_summary.csv', index=False)

# Create visualization
print("\nCreating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Perm 0 vs Mean 6-20
ax = axes[0, 0]
ax.scatter(y_val_target, y_train_target, alpha=0.3, s=20)
ax.plot([0, max(y_val_target.max(), y_train_target.max())],
        [0, max(y_val_target.max(), y_train_target.max())],
        'r--', linewidth=2)
ax.set_xlabel('Mean Count (Perms 6-20)')
ax.set_ylabel('Count (Perm 0)')
ax.set_title(f'Target Correlation: r={r_targets:.3f}')
ax.grid(alpha=0.3)

# Plot 2: Model predictions vs perm 0
ax = axes[0, 1]
ax.scatter(y_test, y_test_pred, alpha=0.3, s=20, label='Test pairs')
ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
ax.set_xlabel('True Count (Perm 0)')
ax.set_ylabel('Predicted Count')
ax.set_title(f'Predictions vs Perm 0: r={r_test_train_target:.3f}')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Model predictions vs mean 6-20
ax = axes[1, 0]
ax.scatter(y_val_test, y_test_pred, alpha=0.3, s=20, label='Test pairs')
ax.plot([0, y_val_test.max()], [0, y_val_test.max()], 'r--', linewidth=2)
ax.set_xlabel('True Mean Count (Perms 6-20)')
ax.set_ylabel('Predicted Count')
ax.set_title(f'Predictions vs Mean 6-20: r={r_test_val_target:.3f}', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
EXPERIMENT 2D: GiGpPW VALIDATION

Training:
  Metapath: GiGpPW
  Train pairs: {len(X_train)}
  Test pairs: {len(X_test)}

Results:
  Target r (perm 0 vs mean 6-20): {r_targets:.4f}
  Model r vs perm 0: {r_test_train_target:.4f}
  Model r vs mean 6-20: {r_test_val_target:.4f}

Status: {final_status}

Success criterion: r > 0.95
Ready for hierarchical: {'YES' if final_status == 'VALIDATED' else 'NO'}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig(results_dir / 'experiment2d_gigppw_plots.png', dpi=150, bbox_inches='tight')

print(f"  Saved: {results_dir / 'experiment2d_gigppw_plots.png'}")

print("\n" + "="*80)
print("EXPERIMENT 2D - GiGpPW COMPLETE")
print("="*80)
print(f"\nFinal Status: {final_status}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2d_gigppw_summary.csv'}")
print(f"  {results_dir / 'experiment2d_gigppw_plots.png'}")
