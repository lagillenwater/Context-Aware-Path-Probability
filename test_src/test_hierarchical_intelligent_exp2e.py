"""
Experiment 2E: Intelligent Degree-Stratified Composition

Key innovation: Only aggregate over genes that ACTUALLY connect,
using ACTUAL CbGiG counts and predicting only edge probabilities.

Approach:
1. Use actual CbGiG counts (accurate, no prediction error)
2. Identify genes that connect to target pathway
3. For each relevant gene, predict edge probability based on degrees
4. Weight actual counts by predicted probabilities

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
print("EXPERIMENT 2E: INTELLIGENT DEGREE-STRATIFIED COMPOSITION")
print("Only Aggregate Over Genes That Actually Connect")
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


print("\n" + "="*80)
print("PHASE 1: Load Perm 0 Data")
print("="*80)

print("\nLoading edges from perm 0...")
CbG_0 = load_edge_matrix('CbG', perm_num=0)
GiG_0 = load_edge_matrix('GiG', perm_num=0)
GpPW_0 = load_edge_matrix('GpPW', perm_num=0)

print(f"  CbG: {CbG_0.shape}, {CbG_0.nnz:,} edges")
print(f"  GiG: {GiG_0.shape}, {GiG_0.nnz:,} edges")
print(f"  GpPW: {GpPW_0.shape}, {GpPW_0.nnz:,} edges")

print("\nComputing CbGiG...")
CbGiG_0 = CbG_0 @ GiG_0
print(f"  CbGiG: {CbGiG_0.shape}, {CbGiG_0.nnz:,} non-zero")

print("\nComputing ground truth CbGiGpPW...")
CbGiGpPW_0 = CbGiG_0 @ GpPW_0
print(f"  CbGiGpPW: {CbGiGpPW_0.shape}, {CbGiGpPW_0.nnz:,} non-zero")


print("\n" + "="*80)
print("PHASE 2: Compute Empirical Edge Probabilities (GpPW)")
print("="*80)

print("\nComputing empirical edge frequencies stratified by degree...")
print("(Using permutations 1-20 for edge probability estimates)")

# Get gene and pathway degrees
gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()

print(f"  Gene degrees: min={gene_degrees.min()}, max={gene_degrees.max()}, n_unique={len(np.unique(gene_degrees))}")
print(f"  Pathway degrees: min={pathway_degrees.min()}, max={pathway_degrees.max()}, n_unique={len(np.unique(pathway_degrees))}")

# Build empirical frequency table: P(edge | deg_gene, deg_pathway)
# For computational efficiency, we'll bin degrees if needed
print("\nBuilding empirical edge frequency table...")

# Count edges by (deg_gene, deg_pathway) across permutations
edge_counts = {}
edge_possible = {}

for perm_num in range(1, 21):
    if perm_num % 5 == 0:
        print(f"  Processing permutation {perm_num}...")

    GpPW_perm = load_edge_matrix('GpPW', perm_num=perm_num)
    if GpPW_perm is None:
        continue

    # For each edge, record its degree pair
    sources, targets = GpPW_perm.nonzero()
    for s, t in zip(sources, targets):
        deg_s = gene_degrees[s]
        deg_t = pathway_degrees[t]
        key = (deg_s, deg_t)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    # Count possible edges (all gene-pathway pairs at each degree combo)
    for deg_g in np.unique(gene_degrees):
        n_genes_with_deg = np.sum(gene_degrees == deg_g)
        for deg_p in np.unique(pathway_degrees):
            n_pathways_with_deg = np.sum(pathway_degrees == deg_p)
            key = (deg_g, deg_p)
            edge_possible[key] = edge_possible.get(key, 0) + (n_genes_with_deg * n_pathways_with_deg)

# Compute frequencies
empirical_freq = {}
for key in edge_counts:
    if key in edge_possible and edge_possible[key] > 0:
        empirical_freq[key] = edge_counts[key] / edge_possible[key]
    else:
        empirical_freq[key] = 0.0

print(f"  Computed frequencies for {len(empirical_freq)} degree pairs")
print(f"  Frequency range: [{min(empirical_freq.values()):.6f}, {max(empirical_freq.values()):.6f}]")


print("\n" + "="*80)
print("PHASE 3: Intelligent Composition for Test Pairs")
print("="*80)

print("\nSampling test pairs...")
np.random.seed(42)

# Sample (Compound, Pathway) pairs
sources_nonzero, targets_nonzero = CbGiGpPW_0.nonzero()
n_nonzero = len(sources_nonzero)

n_nonzero_sample = min(2500, n_nonzero)
idx_nonzero = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
sampled_sources = list(sources_nonzero[idx_nonzero])
sampled_targets = list(targets_nonzero[idx_nonzero])

n_random = 2500
random_sources = np.random.randint(0, CbGiGpPW_0.shape[0], n_random)
random_targets = np.random.randint(0, CbGiGpPW_0.shape[1], n_random)

sampled_sources.extend(random_sources)
sampled_targets.extend(random_targets)

print(f"  Sampled {len(sampled_sources)} pairs (50% non-zero, 50% random)")

print("\nComputing intelligent predictions...")
print("  Strategy: Actual CbGiG counts × Predicted edge probabilities")
print("  Only aggregating over genes that connect to target pathway")

CbGiG_0_lil = CbGiG_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()
CbGiGpPW_0_lil = CbGiGpPW_0.tolil()

y_true = []
y_pred = []
n_intermediates_used = []

for idx, (src, tgt) in enumerate(zip(sampled_sources, sampled_targets)):
    if idx % 1000 == 0:
        print(f"    Processed {idx}/{len(sampled_sources)} pairs...")

    # Ground truth
    true_count = CbGiGpPW_0_lil[src, tgt]

    # Get actual CbGiG counts for all genes
    CbGiG_row = CbGiG_0_lil.getrow(src).toarray().flatten()

    # Get genes that connect to target pathway
    GpPW_col = GpPW_0_csr.getcol(tgt).toarray().flatten()
    genes_connected_to_PW = np.nonzero(GpPW_col)[0]

    # Aggregate only over relevant genes
    predicted_count = 0.0
    n_intermediates = 0

    for gene_idx in genes_connected_to_PW:
        actual_CbGiG_count = CbGiG_row[gene_idx]

        if actual_CbGiG_count == 0:
            continue  # C doesn't reach this gene

        n_intermediates += 1

        # Get degrees
        deg_gene = gene_degrees[gene_idx]
        deg_pw = pathway_degrees[tgt]

        # Look up empirical edge probability
        key = (deg_gene, deg_pw)
        P_edge = empirical_freq.get(key, 0.0)

        # Weight by actual count
        contrib = actual_CbGiG_count * P_edge
        predicted_count += contrib

    y_true.append(true_count)
    y_pred.append(predicted_count)
    n_intermediates_used.append(n_intermediates)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
n_intermediates_used = np.array(n_intermediates_used)

print(f"\n  Computation complete!")
print(f"  Average intermediates per pair: {n_intermediates_used.mean():.1f} (vs 8,611 total genes)")


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
print("PHASE 5: Save Results and Visualizations")
print("="*80)

# Save results
results = {
    'experiment': 'Experiment 2E',
    'description': 'Intelligent degree-stratified composition with actual CbGiG counts',
    'n_pairs': len(y_true),
    'r': r,
    'mae': mae,
    'mean_true': y_true.mean(),
    'mean_pred': y_pred.mean(),
    'avg_intermediates': n_intermediates_used.mean(),
    'status': status
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2e_results.csv', index=False)

# Create visualization
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Predicted vs True
ax = axes[0, 0]
ax.scatter(y_true, y_pred, alpha=0.3, s=20)
max_val = max(y_true.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax.set_xlabel('True Count (CbGiGpPW)')
ax.set_ylabel('Predicted Count')
ax.set_title(f'Intelligent Composition: r={r:.3f}')
ax.grid(alpha=0.3)

# Plot 2: Residuals
ax = axes[0, 1]
residuals = y_true - y_pred
ax.scatter(y_pred, residuals, alpha=0.3, s=20)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Count')
ax.set_ylabel('Residual (True - Predicted)')
ax.set_title('Residual Plot')
ax.grid(alpha=0.3)

# Plot 3: Intermediates used
ax = axes[1, 0]
ax.hist(n_intermediates_used, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(x=n_intermediates_used.mean(), color='r', linestyle='--', linewidth=2,
           label=f'Mean={n_intermediates_used.mean():.1f}')
ax.set_xlabel('Number of Intermediate Genes Used')
ax.set_ylabel('Frequency')
ax.set_title('Sparsity of Aggregation')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
EXPERIMENT 2E: INTELLIGENT COMPOSITION

Approach:
  - Use ACTUAL CbGiG counts (r=0.95)
  - Only aggregate over genes connecting to PW
  - Predict edge probabilities from degrees
  - Weight by actual counts

Results:
  r = {r:.4f}
  MAE = {mae:.4f}

Efficiency:
  Avg intermediates: {n_intermediates_used.mean():.1f}
  vs all genes: 8,611
  Reduction: {100*(1 - n_intermediates_used.mean()/8611):.1f}%

Status: {status}

Comparison to Exp 2B:
  2B (all degrees): r = -0.09 (FAIL)
  2E (intelligent): r = {r:.3f}
"""
ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig(results_dir / 'experiment2e_plots.png', dpi=150, bbox_inches='tight')

print(f"  Saved: {results_dir / 'experiment2e_plots.png'}")

print("\n" + "="*80)
print("EXPERIMENT 2E COMPLETE")
print("="*80)
print(f"\nFinal Status: {status}")
print(f"Correlation: r = {r:.4f}")
print(f"\nKey Innovation: Only aggregate over {n_intermediates_used.mean():.1f} relevant genes")
print(f"  vs {8611} in Experiment 2B")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2e_results.csv'}")
print(f"  {results_dir / 'experiment2e_plots.png'}")
