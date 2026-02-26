"""
Experiment 2D: Validate CbGiG Model with Permutation Generalization

Replicate yesterday's successful r > 0.95 approach for predicting CbGiG counts
from degree features, and validate generalization across permutations.

Design:
- Train on permutation 000
- Test on held-out pairs from permutation 000
- Validate on permutations 5-20

Success criteria:
- Test (perm 000): r > 0.95
- Validation (perms 5-20): average r > 0.90

Date: 2025-11-04
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2D: VALIDATE CbGiG MODEL")
print("Replicating Yesterday's Success + Cross-Permutation Validation")
print("="*80)


def load_edge_matrix(edge_abbrev, perm_num=None):
    """Load edge matrix for specific permutation."""
    if perm_num is None or perm_num == 0:
        edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'
    else:
        edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'

    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    else:
        print(f"  Warning: {edge_file} not found")
        return None


def sample_node_pairs(path_matrix, n_samples=5000, target_ratio=0.5, random_state=42):
    """Sample node pairs for training/testing."""
    np.random.seed(random_state)
    n_source, n_target = path_matrix.shape

    nonzero_sources, nonzero_targets = path_matrix.nonzero()
    n_nonzero = len(nonzero_sources)

    if n_nonzero == 0:
        return np.array([]), np.array([]), np.array([])

    n_nonzero_sample = min(int(n_samples * target_ratio), n_nonzero)
    nonzero_idx = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
    sampled_sources_nz = nonzero_sources[nonzero_idx]
    sampled_targets_nz = nonzero_targets[nonzero_idx]

    n_random_sample = n_samples - n_nonzero_sample
    random_sources = np.random.randint(0, n_source, n_random_sample)
    random_targets = np.random.randint(0, n_target, n_random_sample)

    all_sources = np.concatenate([sampled_sources_nz, random_sources])
    all_targets = np.concatenate([sampled_targets_nz, random_targets])

    path_lil = path_matrix.tolil()
    counts = np.array([
        path_lil[s, t] for s, t in zip(all_sources, all_targets)
    ], dtype=float).flatten()

    return all_sources, all_targets, counts


def extract_features(sources, targets, source_matrix, target_matrix):
    """
    Extract degree features (exactly as in yesterday's successful approach).

    Features (5 total):
    - deg_source, deg_target
    - deg_source × deg_target
    - deg_source², deg_target²
    """
    features = []

    for src, tgt in zip(sources, targets):
        deg_src = source_matrix.getrow(src).nnz
        deg_tgt = target_matrix.getcol(tgt).nnz

        feat = [
            deg_src,
            deg_tgt,
            deg_src * deg_tgt,
            deg_src ** 2,
            deg_tgt ** 2
        ]
        features.append(feat)

    return np.array(features)


print("\n" + "="*80)
print("PHASE 1: Train on Permutation 000")
print("="*80)

print("\nLoading permutation 000...")
CbG_000 = load_edge_matrix('CbG', perm_num=0)
GiG_000 = load_edge_matrix('GiG', perm_num=0)

print(f"  CbG: {CbG_000.shape}, {CbG_000.nnz:,} edges")
print(f"  GiG: {GiG_000.shape}, {GiG_000.nnz:,} edges")

print("\nComputing CbGiG for permutation 000...")
t0 = time.time()
CbGiG_000 = CbG_000 @ GiG_000
t_compute = time.time() - t0

print(f"  CbGiG: {CbGiG_000.shape}, {CbGiG_000.nnz:,} non-zero")
print(f"  Computation time: {t_compute:.3f}s")

print("\nSampling training pairs...")
sources_train, targets_train, y_train = sample_node_pairs(
    CbGiG_000, n_samples=10000, target_ratio=0.5, random_state=42
)

print(f"  Training: {len(y_train)} pairs")
print(f"  Non-zero: {np.sum(y_train > 0)}")
print(f"  Range: [{y_train.min():.1f}, {y_train.max():.1f}]")

print("\nExtracting features...")
X_train = extract_features(sources_train, targets_train, CbG_000, GiG_000)
print(f"  Feature matrix: {X_train.shape}")

print("\nTraining model...")
t0 = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
t_train = time.time() - t0

y_train_pred = model.predict(X_train)
r_train = np.corrcoef(y_train, y_train_pred)[0, 1]

print(f"  Training time: {t_train:.4f}s")
print(f"  Training r: {r_train:.4f}")

print("\n" + "="*80)
print("PHASE 2: Test on Permutation 000 (Held-Out Pairs)")
print("="*80)

print("\nSampling test pairs...")
sources_test, targets_test, y_test = sample_node_pairs(
    CbGiG_000, n_samples=5000, target_ratio=0.5, random_state=123
)

print(f"  Test: {len(y_test)} pairs")
print(f"  Non-zero: {np.sum(y_test > 0)}")

print("\nExtracting features and predicting...")
X_test = extract_features(sources_test, targets_test, CbG_000, GiG_000)
y_test_pred = model.predict(X_test)

r_test = np.corrcoef(y_test, y_test_pred)[0, 1]
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"\nTest Performance (Permutation 000):")
print(f"  r = {r_test:.4f}")
print(f"  R² = {r2_test:.4f}")
print(f"  MAE = {mae_test:.4f}")

if r_test > 0.95:
    print(f"  Status: SUCCESS (r > 0.95)")
elif r_test > 0.90:
    print(f"  Status: ACCEPTABLE (0.90 < r < 0.95)")
else:
    print(f"  Status: FAILURE (r < 0.90)")
    print(f"  WARNING: Model does not replicate yesterday's performance!")

print("\n" + "="*80)
print("PHASE 3: Validate on Permutations 5-20")
print("="*80)

print("\nTesting generalization across permutations...")

validation_results = []
validation_perms = range(5, 21)

for perm_num in validation_perms:
    print(f"\n  Permutation {perm_num:03d}:")

    # Load permuted edges
    CbG_perm = load_edge_matrix('CbG', perm_num=perm_num)
    GiG_perm = load_edge_matrix('GiG', perm_num=perm_num)

    if CbG_perm is None or GiG_perm is None:
        print(f"    Skipping (files not found)")
        continue

    # Compute CbGiG for this permutation
    CbGiG_perm = CbG_perm @ GiG_perm

    # Sample pairs
    sources_val, targets_val, y_val = sample_node_pairs(
        CbGiG_perm, n_samples=5000, target_ratio=0.5, random_state=perm_num
    )

    if len(y_val) == 0:
        print(f"    Skipping (no valid pairs)")
        continue

    # Extract features and predict
    X_val = extract_features(sources_val, targets_val, CbG_perm, GiG_perm)
    y_val_pred = model.predict(X_val)

    r_val = np.corrcoef(y_val, y_val_pred)[0, 1]
    r2_val = r2_score(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)

    print(f"    r = {r_val:.4f}, R² = {r2_val:.4f}, MAE = {mae_val:.4f}")

    validation_results.append({
        'permutation': perm_num,
        'r': r_val,
        'r2': r2_val,
        'mae': mae_val
    })

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

df_val = pd.DataFrame(validation_results)

avg_r = df_val['r'].mean()
std_r = df_val['r'].std()
min_r = df_val['r'].min()
max_r = df_val['r'].max()

print(f"\nCross-Permutation Performance:")
print(f"  Average r: {avg_r:.4f} ± {std_r:.4f}")
print(f"  Range: [{min_r:.4f}, {max_r:.4f}]")
print(f"  N permutations: {len(validation_results)}")

if avg_r > 0.90:
    val_status = "SUCCESS"
    print(f"\n  Status: SUCCESS (avg r > 0.90)")
elif avg_r > 0.80:
    val_status = "PARTIAL"
    print(f"\n  Status: PARTIAL (0.80 < avg r < 0.90)")
else:
    val_status = "FAILURE"
    print(f"\n  Status: FAILURE (avg r < 0.80)")

print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

print(f"\nIn-Distribution (Perm 000 test): r = {r_test:.4f}")
print(f"Cross-Permutation (Perms 5-20): avg r = {avg_r:.4f} ± {std_r:.4f}")

if r_test > 0.95 and avg_r > 0.90:
    final_status = "VALIDATED"
    print(f"\nFINAL STATUS: VALIDATED")
    print(f"  Model replicates yesterday's success (r > 0.95)")
    print(f"  Model generalizes across permutations (avg r > 0.90)")
    print(f"  Ready to proceed with hierarchical composition experiments")
elif r_test > 0.90:
    final_status = "ACCEPTABLE"
    print(f"\nFINAL STATUS: ACCEPTABLE")
    print(f"  Model performs well but below yesterday's r > 0.95")
    print(f"  May proceed with caution")
else:
    final_status = "FAILED"
    print(f"\nFINAL STATUS: FAILED")
    print(f"  Model does not replicate yesterday's performance")
    print(f"  Should not proceed with hierarchical experiments until resolved")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'experiment': 'Experiment 2D',
    'description': 'Validate CbGiG model with cross-permutation testing',
    'metapath': 'CbGiG',
    'train_perm': 0,
    'n_train': len(y_train),
    'n_test': len(y_test),
    'r_train': r_train,
    'r_test': r_test,
    'r2_test': r2_test,
    'mae_test': mae_test,
    'val_avg_r': avg_r,
    'val_std_r': std_r,
    'val_min_r': min_r,
    'val_max_r': max_r,
    'n_val_perms': len(validation_results),
    'test_status': 'SUCCESS' if r_test > 0.95 else 'ACCEPTABLE' if r_test > 0.90 else 'FAILURE',
    'val_status': val_status,
    'final_status': final_status
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2d_validation_summary.csv', index=False)

df_val.to_csv(results_dir / 'experiment2d_validation_details.csv', index=False)

# Create visualization
print("\nCreating visualizations...")

fig = plt.figure(figsize=(15, 10))

# Plot 1: Test predictions (perm 000)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_test_pred, alpha=0.3, s=20)
ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
ax1.set_xlabel('True Count')
ax1.set_ylabel('Predicted Count')
ax1.set_title(f'Test (Perm 000): r={r_test:.3f}')
ax1.grid(alpha=0.3)

# Plot 2: Validation correlations
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df_val['permutation'], df_val['r'], 'o-', linewidth=2, markersize=6)
ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='r=0.95')
ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='r=0.90')
ax2.axhline(y=avg_r, color='blue', linestyle='-', linewidth=2, label=f'Mean={avg_r:.3f}')
ax2.set_xlabel('Permutation')
ax2.set_ylabel('Correlation (r)')
ax2.set_title('Cross-Permutation Validation')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Distribution of validation correlations
ax3 = plt.subplot(2, 3, 3)
ax3.hist(df_val['r'], bins=15, edgecolor='black', alpha=0.7)
ax3.axvline(x=avg_r, color='blue', linestyle='--', linewidth=2, label=f'Mean={avg_r:.3f}')
ax3.axvline(x=0.95, color='green', linestyle='--', linewidth=2, label='Target=0.95')
ax3.set_xlabel('Correlation (r)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Validation r')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: MAE across permutations
ax4 = plt.subplot(2, 3, 4)
ax4.plot(df_val['permutation'], df_val['mae'], 'o-', linewidth=2, markersize=6)
ax4.axhline(y=df_val['mae'].mean(), color='blue', linestyle='--', linewidth=2,
            label=f'Mean={df_val["mae"].mean():.3f}')
ax4.set_xlabel('Permutation')
ax4.set_ylabel('MAE')
ax4.set_title('MAE Across Permutations')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Test residuals
ax5 = plt.subplot(2, 3, 5)
residuals = y_test - y_test_pred
ax5.scatter(y_test_pred, residuals, alpha=0.3, s=20)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Count')
ax5.set_ylabel('Residual')
ax5.set_title('Test Residuals (Perm 000)')
ax5.grid(alpha=0.3)

# Plot 6: Summary status
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
EXPERIMENT 2D: VALIDATION RESULTS

In-Distribution (Perm 000):
  r = {r_test:.4f}
  Status: {'SUCCESS' if r_test > 0.95 else 'ACCEPTABLE' if r_test > 0.90 else 'FAILURE'}

Cross-Permutation (Perms 5-20):
  Average r = {avg_r:.4f} ± {std_r:.4f}
  Range: [{min_r:.4f}, {max_r:.4f}]
  Status: {val_status}

FINAL STATUS: {final_status}

Ready for hierarchical experiments: {'YES' if final_status == 'VALIDATED' else 'NO'}
"""
ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig(results_dir / 'experiment2d_validation_plots.png', dpi=150, bbox_inches='tight')

print(f"  Saved: {results_dir / 'experiment2d_validation_plots.png'}")

print("\n" + "="*80)
print("EXPERIMENT 2D COMPLETE")
print("="*80)
print(f"\nFinal Status: {final_status}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2d_validation_summary.csv'}")
print(f"  {results_dir / 'experiment2d_validation_details.csv'}")
print(f"  {results_dir / 'experiment2d_validation_plots.png'}")
