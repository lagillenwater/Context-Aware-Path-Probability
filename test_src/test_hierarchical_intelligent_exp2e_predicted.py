"""
Experiment 2E (Updated): Intelligent Composition with PREDICTED CbGiG

Key changes from original 2E:
- Use PREDICTED CbGiG counts from validated model (r=0.95)
- Use empirical edge frequencies for GpPW
- Only aggregate over genes that actually connect to target

This tests if hierarchical composition works when using the validated
base models instead of ground truth counts.

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
print("EXPERIMENT 2E (UPDATED): INTELLIGENT COMPOSITION WITH PREDICTED CbGiG")
print("Using validated CbGiG model (r=0.95) + empirical GpPW frequencies")
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


def extract_degree_features(deg_u, deg_v):
    """Extract 5 degree features for prediction."""
    return np.array([
        deg_u, deg_v, deg_u * deg_v, deg_u ** 2, deg_v ** 2
    ], dtype=np.float64)


print("\n" + "="*80)
print("PHASE 1: Load Data and Train CbGiG Model")
print("="*80)

print("\nLoading edges from perm 0...")
CbG_0 = load_edge_matrix('CbG', perm_num=0)
GiG_0 = load_edge_matrix('GiG', perm_num=0)
GpPW_0 = load_edge_matrix('GpPW', perm_num=0)

print(f"  CbG: {CbG_0.shape}, {CbG_0.nnz:,} edges")
print(f"  GiG: {GiG_0.shape}, {GiG_0.nnz:,} edges")
print(f"  GpPW: {GpPW_0.shape}, {GpPW_0.nnz:,} edges")

print("\nComputing CbGiG for perm 0...")
CbGiG_0 = CbG_0 @ GiG_0
print(f"  CbGiG: {CbGiG_0.shape}, {CbGiG_0.nnz:,} non-zero")

print("\nComputing ground truth CbGiGpPW...")
CbGiGpPW_0 = CbGiG_0 @ GpPW_0
print(f"  CbGiGpPW: {CbGiGpPW_0.shape}, {CbGiGpPW_0.nnz:,} non-zero")

# Get node degrees
compound_degrees = np.array(CbG_0.sum(axis=1)).flatten()
gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()

print(f"\nNode degrees:")
print(f"  Compounds: min={compound_degrees.min()}, max={compound_degrees.max()}")
print(f"  Genes: min={gene_degrees.min()}, max={gene_degrees.max()}")
print(f"  Pathways: min={pathway_degrees.min()}, max={pathway_degrees.max()}")

# Train CbGiG model (replicating Experiment 2D methodology)
print("\n" + "-"*80)
print("Training CbGiG model...")
print("-"*80)

print("\nSampling pairs for CbGiG training...")
np.random.seed(42)
sources_nonzero, targets_nonzero = CbGiG_0.nonzero()
n_nonzero = len(sources_nonzero)

n_nonzero_sample = min(5000, n_nonzero)
idx_nonzero = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
sampled_sources = list(sources_nonzero[idx_nonzero])
sampled_targets = list(targets_nonzero[idx_nonzero])

n_random = 5000
random_sources = np.random.randint(0, CbGiG_0.shape[0], n_random)
random_targets = np.random.randint(0, CbGiG_0.shape[1], n_random)

sampled_sources.extend(random_sources)
sampled_targets.extend(random_targets)

print(f"  Sampled {len(sampled_sources)} pairs for training")

# Extract features and targets
print("\nExtracting features and computing targets...")
X_cbgig = []
y_train_cbgig = []

CbGiG_0_lil = CbGiG_0.tolil()

for src, tgt in zip(sampled_sources, sampled_targets):
    deg_c = compound_degrees[src]
    deg_g = gene_degrees[tgt]
    features = extract_degree_features(deg_c, deg_g)
    X_cbgig.append(features)
    y_train_cbgig.append(CbGiG_0_lil[src, tgt])

X_cbgig = np.array(X_cbgig)
y_train_cbgig = np.array(y_train_cbgig, dtype=float).flatten()

print(f"  Features: {X_cbgig.shape}")
print(f"  Target range: [{y_train_cbgig.min():.1f}, {y_train_cbgig.max():.1f}], mean={y_train_cbgig.mean():.3f}")

# Load validation targets from perms 6-20
print("\nLoading validation targets from perms 6-20...")
val_perm_counts = []

for perm_num in range(6, 21):
    if perm_num % 5 == 0:
        print(f"  Processing permutation {perm_num}...")

    CbG_perm = load_edge_matrix('CbG', perm_num=perm_num)
    GiG_perm = load_edge_matrix('GiG', perm_num=perm_num)

    if CbG_perm is None or GiG_perm is None:
        continue

    CbGiG_perm = CbG_perm @ GiG_perm
    CbGiG_perm_lil = CbGiG_perm.tolil()

    counts = np.array([
        CbGiG_perm_lil[src, tgt] for src, tgt in zip(sampled_sources, sampled_targets)
    ], dtype=float).flatten()

    val_perm_counts.append(counts)

y_val_cbgig = np.mean(val_perm_counts, axis=0)
print(f"  Validation target: mean={y_val_cbgig.mean():.3f}")

# Train model
print("\nTraining CbGiG model...")
X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
    X_cbgig, y_train_cbgig, y_val_cbgig, test_size=0.2, random_state=42
)

model_CbGiG = LinearRegression()
model_CbGiG.fit(X_train, y_val_train)

y_pred_test = model_CbGiG.predict(X_test)
r_cbgig = pearsonr(y_pred_test, y_val_test)[0]

print(f"  CbGiG model r vs mean 6-20: {r_cbgig:.4f}")
if r_cbgig > 0.95:
    print(f"  Status: VALIDATED")
else:
    print(f"  WARNING: Model performance below threshold (expected r > 0.95)")


print("\n" + "="*80)
print("PHASE 2: Compute Empirical Edge Probabilities (GpPW)")
print("="*80)

print("\nBuilding empirical edge frequency table for GpPW...")
print("(Using permutations 1-20)")

edge_counts = {}
edge_possible = {}

for perm_num in range(1, 21):
    if perm_num % 5 == 0:
        print(f"  Processing permutation {perm_num}...")

    GpPW_perm = load_edge_matrix('GpPW', perm_num=perm_num)
    if GpPW_perm is None:
        continue

    sources, targets = GpPW_perm.nonzero()
    for s, t in zip(sources, targets):
        deg_s = gene_degrees[s]
        deg_t = pathway_degrees[t]
        key = (deg_s, deg_t)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    for deg_g in np.unique(gene_degrees):
        n_genes_with_deg = np.sum(gene_degrees == deg_g)
        for deg_p in np.unique(pathway_degrees):
            n_pathways_with_deg = np.sum(pathway_degrees == deg_p)
            key = (deg_g, deg_p)
            edge_possible[key] = edge_possible.get(key, 0) + (n_genes_with_deg * n_pathways_with_deg)

empirical_freq = {}
for key in edge_counts:
    if key in edge_possible and edge_possible[key] > 0:
        empirical_freq[key] = edge_counts[key] / edge_possible[key]
    else:
        empirical_freq[key] = 0.0

print(f"  Computed frequencies for {len(empirical_freq)} degree pairs")
print(f"  Frequency range: [{min(empirical_freq.values()):.6f}, {max(empirical_freq.values()):.6f}]")


print("\n" + "="*80)
print("PHASE 3: Intelligent Composition with PREDICTED CbGiG")
print("="*80)

print("\nSampling test pairs for CbGiGpPW...")
np.random.seed(123)

sources_nonzero_pw, targets_nonzero_pw = CbGiGpPW_0.nonzero()
n_nonzero_pw = len(sources_nonzero_pw)

n_nonzero_sample_pw = min(2500, n_nonzero_pw)
idx_nonzero_pw = np.random.choice(n_nonzero_pw, n_nonzero_sample_pw, replace=False)
sampled_sources_pw = list(sources_nonzero_pw[idx_nonzero_pw])
sampled_targets_pw = list(targets_nonzero_pw[idx_nonzero_pw])

n_random_pw = 2500
random_sources_pw = np.random.randint(0, CbGiGpPW_0.shape[0], n_random_pw)
random_targets_pw = np.random.randint(0, CbGiGpPW_0.shape[1], n_random_pw)

sampled_sources_pw.extend(random_sources_pw)
sampled_targets_pw.extend(random_targets_pw)

print(f"  Sampled {len(sampled_sources_pw)} pairs")

print("\nComputing predictions using intelligent composition...")
print("  PREDICTED CbGiG (from model) × Empirical GpPW frequencies")
print("  Only aggregating over genes connecting to target pathway")

GpPW_0_csr = GpPW_0.tocsr()
CbGiGpPW_0_lil = CbGiGpPW_0.tolil()

y_true = []
y_pred = []
n_intermediates_used = []
predictions_breakdown = []

for idx, (src, tgt) in enumerate(zip(sampled_sources_pw, sampled_targets_pw)):
    if idx % 1000 == 0:
        print(f"    Processed {idx}/{len(sampled_sources_pw)} pairs...")

    # Ground truth
    true_count = CbGiGpPW_0_lil[src, tgt]

    # Get genes that connect to target pathway
    GpPW_col = GpPW_0_csr.getcol(tgt).toarray().flatten()
    genes_connected_to_PW = np.nonzero(GpPW_col)[0]

    # Aggregate using PREDICTED CbGiG counts
    deg_c = compound_degrees[src]
    deg_pw = pathway_degrees[tgt]

    predicted_count = 0.0
    n_intermediates = 0
    sum_predicted_cbgig = 0.0
    sum_gpw_freq = 0.0

    for gene_idx in genes_connected_to_PW:
        deg_g = gene_degrees[gene_idx]

        # Predict CbGiG count using trained model
        features = extract_degree_features(deg_c, deg_g)
        predicted_CbGiG = model_CbGiG.predict([features])[0]

        if predicted_CbGiG <= 0:
            continue

        n_intermediates += 1
        sum_predicted_cbgig += predicted_CbGiG

        # Get empirical GpPW frequency
        key = (deg_g, deg_pw)
        P_edge_GpPW = empirical_freq.get(key, 0.0)
        sum_gpw_freq += P_edge_GpPW

        # Compose
        contrib = predicted_CbGiG * P_edge_GpPW
        predicted_count += contrib

    y_true.append(true_count)
    y_pred.append(predicted_count)
    n_intermediates_used.append(n_intermediates)

    # Store breakdown for first 100 pairs for investigation
    if idx < 100:
        predictions_breakdown.append({
            'pair_idx': idx,
            'true': true_count,
            'predicted': predicted_count,
            'n_intermediates': n_intermediates,
            'deg_C': deg_c,
            'deg_PW': deg_pw,
            'sum_predicted_cbgig': sum_predicted_cbgig,
            'avg_gpw_freq': sum_gpw_freq / n_intermediates if n_intermediates > 0 else 0
        })

y_true = np.array(y_true)
y_pred = np.array(y_pred)
n_intermediates_used = np.array(n_intermediates_used)

print(f"\n  Computation complete!")
print(f"  Average intermediates per pair: {n_intermediates_used.mean():.1f}")


print("\n" + "="*80)
print("PHASE 4: Evaluate Performance")
print("="*80)

r = pearsonr(y_true, y_pred)[0]
mae = np.mean(np.abs(y_true - y_pred))

print(f"\nPerformance Metrics:")
print(f"  Correlation (r): {r:.4f}")
print(f"  MAE: {mae:.4f}")

print(f"\nGround Truth Distribution:")
print(f"  Min: {y_true.min():.2f}")
print(f"  Max: {y_true.max():.2f}")
print(f"  Mean: {y_true.mean():.2f}")
print(f"  Non-zero: {np.sum(y_true > 0)} / {len(y_true)}")

print(f"\nPrediction Distribution:")
print(f"  Min: {y_pred.min():.2f}")
print(f"  Max: {y_pred.max():.2f}")
print(f"  Mean: {y_pred.mean():.2f}")
print(f"  Non-zero: {np.sum(y_pred > 0)} / {len(y_pred)}")

# Compute prediction ratio
ratio = y_pred.mean() / y_true.mean() if y_true.mean() > 0 else 0
print(f"\nPrediction Ratio (mean_pred / mean_true): {ratio:.3f}")

if r > 0.95:
    status = "SUCCESS"
    print(f"\n  Status: SUCCESS (r > 0.95)")
elif r > 0.80:
    status = "PROMISING"
    print(f"\n  Status: PROMISING (r > 0.80)")
elif r > 0.50:
    status = "PARTIAL"
    print(f"\n  Status: PARTIAL (r > 0.50)")
else:
    status = "FAILURE"
    print(f"\n  Status: FAILURE (r < 0.50)")


print("\n" + "="*80)
print("PHASE 5: Investigate Predictions")
print("="*80)

print("\nInvestigating why predictions differ from ground truth...")

# Analyze by prediction magnitude
bins = [0, 0.1, 1, 10, 100, 1000]
bin_labels = ['[0, 0.1)', '[0.1, 1)', '[1, 10)', '[10, 100)', '[100+']
y_pred_binned = np.digitize(y_pred, bins)

print("\nPerformance by prediction magnitude:")
for i in range(1, len(bins)):
    mask = y_pred_binned == i
    if np.sum(mask) == 0:
        continue

    r_bin = pearsonr(y_true[mask], y_pred[mask])[0] if np.sum(mask) > 1 else np.nan
    mean_true_bin = y_true[mask].mean()
    mean_pred_bin = y_pred[mask].mean()

    print(f"  {bin_labels[i-1]}: n={np.sum(mask)}, r={r_bin:.3f}, mean_true={mean_true_bin:.2f}, mean_pred={mean_pred_bin:.2f}")

# Analyze sample of predictions
print("\nSample prediction breakdown (first 10 pairs):")
df_breakdown = pd.DataFrame(predictions_breakdown[:10])
print(df_breakdown.to_string(index=False))

# Investigate bias
residuals = y_true - y_pred
print(f"\nResidual Analysis:")
print(f"  Mean residual: {residuals.mean():.3f} (positive = underprediction)")
print(f"  Median residual: {np.median(residuals):.3f}")
print(f"  Std residual: {residuals.std():.3f}")

# Check if bias is systematic
mask_nonzero = y_true > 0
if np.sum(mask_nonzero) > 0:
    print(f"\nFor non-zero ground truth pairs:")
    print(f"  Mean true: {y_true[mask_nonzero].mean():.3f}")
    print(f"  Mean predicted: {y_pred[mask_nonzero].mean():.3f}")
    print(f"  Ratio: {y_pred[mask_nonzero].mean() / y_true[mask_nonzero].mean():.3f}")


print("\n" + "="*80)
print("PHASE 6: Save Results and Visualizations")
print("="*80)

results = {
    'experiment': 'Experiment 2E (Updated)',
    'description': 'Intelligent composition with PREDICTED CbGiG + empirical GpPW',
    'n_pairs': len(y_true),
    'r': r,
    'mae': mae,
    'mean_true': y_true.mean(),
    'mean_pred': y_pred.mean(),
    'ratio_pred_true': ratio,
    'avg_intermediates': n_intermediates_used.mean(),
    'r_cbgig_model': r_cbgig,
    'status': status
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2e_predicted_results.csv', index=False)

# Save breakdown for investigation
df_breakdown_full = pd.DataFrame(predictions_breakdown)
df_breakdown_full.to_csv(results_dir / 'experiment2e_predicted_breakdown.csv', index=False)

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Predicted vs True
ax = axes[0, 0]
ax.scatter(y_true, y_pred, alpha=0.3, s=20)
max_val = max(y_true.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax.set_xlabel('True Count')
ax.set_ylabel('Predicted Count')
ax.set_title(f'Predicted CbGiG: r={r:.3f}')
ax.grid(alpha=0.3)

# Plot 2: Residuals
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.3, s=20)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Count')
ax.set_ylabel('Residual (True - Predicted)')
ax.set_title('Residual Plot')
ax.grid(alpha=0.3)

# Plot 3: Log scale
ax = axes[0, 2]
mask_both_nonzero = (y_true > 0) & (y_pred > 0)
if np.sum(mask_both_nonzero) > 10:
    ax.scatter(np.log10(y_true[mask_both_nonzero] + 1),
               np.log10(y_pred[mask_both_nonzero] + 1), alpha=0.3, s=20)
    max_log = max(np.log10(y_true[mask_both_nonzero] + 1).max(),
                   np.log10(y_pred[mask_both_nonzero] + 1).max())
    ax.plot([0, max_log], [0, max_log], 'r--', linewidth=2)
ax.set_xlabel('Log10(True + 1)')
ax.set_ylabel('Log10(Predicted + 1)')
ax.set_title('Log Scale Comparison')
ax.grid(alpha=0.3)

# Plot 4: Intermediates distribution
ax = axes[1, 0]
ax.hist(n_intermediates_used, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(x=n_intermediates_used.mean(), color='r', linestyle='--', linewidth=2,
           label=f'Mean={n_intermediates_used.mean():.1f}')
ax.set_xlabel('Number of Intermediate Genes')
ax.set_ylabel('Frequency')
ax.set_title('Aggregation Sparsity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Prediction by magnitude
ax = axes[1, 1]
bin_means_true = []
bin_means_pred = []
bin_centers = []
for i in range(1, len(bins)):
    mask = y_pred_binned == i
    if np.sum(mask) > 0:
        bin_means_true.append(y_true[mask].mean())
        bin_means_pred.append(y_pred[mask].mean())
        bin_centers.append(i)

if len(bin_centers) > 0:
    ax.plot(bin_centers, bin_means_true, 'o-', label='True', linewidth=2, markersize=8)
    ax.plot(bin_centers, bin_means_pred, 's-', label='Predicted', linewidth=2, markersize=8)
    ax.set_xticks(range(1, len(bins)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
ax.set_ylabel('Mean Count')
ax.set_title('Mean by Prediction Magnitude')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Summary
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
EXP 2E (UPDATED): PREDICTED CbGiG

Approach:
  - PREDICTED CbGiG (model r={r_cbgig:.3f})
  - Empirical GpPW frequencies
  - Aggregate over connected genes only

Results:
  r = {r:.4f}
  MAE = {mae:.4f}
  Mean true: {y_true.mean():.2f}
  Mean pred: {y_pred.mean():.2f}
  Ratio: {ratio:.3f}

Efficiency:
  Avg intermediates: {n_intermediates_used.mean():.1f}

Status: {status}

Comparison:
  2B (all degrees): r = -0.09
  2E (actual counts): r = 0.855
  2E (predicted): r = {r:.3f}
"""
ax.text(0.05, 0.5, summary_text, fontsize=8, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig(results_dir / 'experiment2e_predicted_plots.png', dpi=150, bbox_inches='tight')

print(f"  Saved: {results_dir / 'experiment2e_predicted_plots.png'}")

print("\n" + "="*80)
print("EXPERIMENT 2E (UPDATED) COMPLETE")
print("="*80)
print(f"\nFinal Status: {status}")
print(f"Correlation: r = {r:.4f}")
print(f"Prediction ratio: {ratio:.3f}")
print(f"\nComparison to original 2E:")
print(f"  2E with actual CbGiG: r = 0.855")
print(f"  2E with predicted CbGiG: r = {r:.3f}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2e_predicted_results.csv'}")
print(f"  {results_dir / 'experiment2e_predicted_breakdown.csv'}")
print(f"  {results_dir / 'experiment2e_predicted_plots.png'}")
