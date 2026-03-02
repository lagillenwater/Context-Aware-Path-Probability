"""
Hierarchical Path Prediction - Experiment 2C

Predicted Counts × Edge Probabilities (Two Variants)

Fix for Experiment 2B's overestimation: Use PROBABILITIES instead of COUNTS
for the second term.

Approach:
contrib[d] = predicted_CbGiG_to_deg_d × P(gene_deg_d → PW_deg_pw)

Two variants:
- 2C-v1: Analytical edge probabilities (configuration model)
- 2C-v2: Empirical frequencies from actual graph

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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("HIERARCHICAL PATH PREDICTION - EXPERIMENT 2C")
print("Predicted Counts × Edge Probabilities")
print("="*80)


def load_edge_matrix(edge_abbrev):
    """Load edge matrix."""
    edge_file = data_dir / 'edges' / f'{edge_abbrev}.sparse.npz'
    if edge_file.exists():
        return sp.load_npz(str(edge_file))
    raise FileNotFoundError(f"Could not find edge file for {edge_abbrev}")


def sample_node_pairs_for_edge(edge_matrix, n_samples=10000, target_ratio=0.5,
                                random_state=42):
    """Sample node pairs for training a single edge model."""
    np.random.seed(random_state)
    n_source, n_target = edge_matrix.shape

    nonzero_sources, nonzero_targets = edge_matrix.nonzero()
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

    edge_lil = edge_matrix.tolil()
    counts = np.array([
        edge_lil[s, t] for s, t in zip(all_sources, all_targets)
    ], dtype=float).flatten()

    return all_sources, all_targets, counts


def extract_degree_features_for_pair(src, tgt, source_matrix, target_matrix):
    """Extract degree-based features for a single pair."""
    deg_src = source_matrix.getrow(src).nnz
    deg_tgt = target_matrix.getcol(tgt).nnz

    log_deg_src = np.log1p(deg_src)
    log_deg_tgt = np.log1p(deg_tgt)

    features = [
        deg_src,
        deg_tgt,
        log_deg_src,
        log_deg_tgt,
        deg_src * deg_tgt,
        log_deg_src * log_deg_tgt,
        deg_src ** 2,
        deg_tgt ** 2
    ]

    return features


def compute_analytical_edge_probability(deg_source, deg_target, n_edges,
                                        n_source_nodes, n_target_nodes):
    """
    Compute analytical edge probability using configuration model.

    For bipartite graph, P(edge) ≈ (deg_u * deg_v) / (n_edges)
    """
    if deg_source == 0 or deg_target == 0:
        return 0.0

    # Configuration model approximation
    prob = (deg_source * deg_target) / (n_edges * n_source_nodes)

    # Clip to [0, 1]
    return min(prob, 1.0)


def compute_empirical_edge_probabilities(edge_matrix):
    """
    Compute empirical P(node_deg_d connects to target_deg_d') for all degree pairs.

    Returns dict: (source_degree, target_degree) -> probability
    """
    print("\n  Computing empirical edge probabilities...")

    # Get degrees
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    # Group nodes by degree
    source_by_degree = defaultdict(list)
    for i, deg in enumerate(source_degrees):
        if deg > 0:
            source_by_degree[int(deg)].append(i)

    target_by_degree = defaultdict(list)
    for j, deg in enumerate(target_degrees):
        if deg > 0:
            target_by_degree[int(deg)].append(j)

    # Compute empirical probabilities
    empirical_probs = {}

    for deg_src in source_by_degree:
        sources = source_by_degree[deg_src]

        for deg_tgt in target_by_degree:
            targets = target_by_degree[deg_tgt]

            # Count connections
            n_connections = 0
            n_possible = len(sources) * len(targets)

            for src in sources:
                connected_targets = edge_matrix.getrow(src).nonzero()[1]
                connected_in_degree_class = len([t for t in connected_targets
                                                 if t in targets])
                n_connections += connected_in_degree_class

            # Probability
            empirical_probs[(deg_src, deg_tgt)] = n_connections / n_possible if n_possible > 0 else 0.0

    print(f"    Computed probabilities for {len(empirical_probs)} degree pairs")

    return empirical_probs


print("\n" + "="*80)
print("STEP 1: Load Data and Train Subpath Models (Reuse from 2B)")
print("="*80)

print("\nLoading edge matrices...")
CbG = load_edge_matrix('CbG')
GiG = load_edge_matrix('GiG')
GpPW = load_edge_matrix('GpPW')

print(f"  CbG: {CbG.shape}, {CbG.nnz:,} edges")
print(f"  GiG: {GiG.shape}, {GiG.nnz:,} edges")
print(f"  GpPW: {GpPW.shape}, {GpPW.nnz:,} edges")

print("\nComputing subpath counts...")
CbGiG = CbG @ GiG
GiGpPW = GiG @ GpPW
CbGiGpPW = CbGiG @ GpPW

print(f"  CbGiG: {CbGiG.nnz:,} non-zero")
print(f"  GiGpPW: {GiGpPW.nnz:,} non-zero")
print(f"  CbGiGpPW: {CbGiGpPW.nnz:,} non-zero")

# Get gene degrees
gene_degrees_in_GiG = np.array(GiG.sum(axis=1)).flatten()
gene_degrees_in_GpPW = np.array(GpPW.sum(axis=0)).flatten()  # Gene degrees in GpPW
max_gene_degree = int(max(gene_degrees_in_GiG.max(), gene_degrees_in_GpPW.max()))

print(f"  Max gene degree: {max_gene_degree}")

# Train Model 1: CbGiG
print("\nTraining Model 1: CbGiG from (deg_C, deg_G2)...")
src_1, tgt_1, y_1 = sample_node_pairs_for_edge(
    CbGiG, n_samples=10000, target_ratio=0.5, random_state=42
)

X_1_list = []
for src, tgt in zip(src_1, tgt_1):
    features = extract_degree_features_for_pair(src, tgt, CbG, GiG)
    X_1_list.append(features)
X_1 = np.array(X_1_list)

model_CbGiG = LinearRegression()
model_CbGiG.fit(X_1, y_1)

y_1_pred = model_CbGiG.predict(X_1)
r_1 = np.corrcoef(y_1, y_1_pred)[0, 1]
print(f"  Training r: {r_1:.4f}")

print("\n" + "="*80)
print("STEP 2: Compute Edge Probabilities (Two Variants)")
print("="*80)

# Variant 1: Analytical probabilities
print("\nVariant 1: Analytical Edge Probabilities")
n_genes = GpPW.shape[0]
n_pathways = GpPW.shape[1]
n_edges_GpPW = GpPW.nnz

print(f"  GpPW network: {n_genes} genes, {n_pathways} pathways, {n_edges_GpPW:,} edges")

# Variant 2: Empirical probabilities
print("\nVariant 2: Empirical Edge Probabilities")
empirical_probs_GpPW = compute_empirical_edge_probabilities(GpPW)

print("\n" + "="*80)
print("STEP 3: Make Predictions (Both Variants)")
print("="*80)

print("\nSampling test pairs...")
sources_test, targets_test, y_test = sample_node_pairs_for_edge(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=45
)

print(f"  Test: {len(y_test)} pairs, {np.sum(y_test > 0)} with paths")

# Prepare all gene degrees for vectorization
all_gene_degrees = np.arange(1, max_gene_degree + 1)
n_deg = len(all_gene_degrees)

print(f"  Will iterate over {n_deg} gene degree values")

# Variant 1: Analytical
print("\nVariant 1 - Computing predictions with analytical probabilities...")
t0 = time.time()

predictions_analytical = []
for i, (src, tgt) in enumerate(zip(sources_test, targets_test)):
    if (i + 1) % 1000 == 0:
        print(f"    Processed {i+1}/{len(sources_test)} pairs...")

    deg_C = CbG.getrow(src).nnz
    deg_PW = GpPW.getcol(tgt).nnz

    # Vectorized feature construction for CbGiG
    features_CbGiG = np.column_stack([
        np.full(n_deg, deg_C),
        all_gene_degrees,
        np.full(n_deg, np.log1p(deg_C)),
        np.log1p(all_gene_degrees),
        np.full(n_deg, deg_C) * all_gene_degrees,
        np.full(n_deg, np.log1p(deg_C)) * np.log1p(all_gene_degrees),
        np.full(n_deg, deg_C ** 2),
        all_gene_degrees ** 2
    ])

    # Predict CbGiG counts
    predicted_CbGiG = model_CbGiG.predict(features_CbGiG)
    predicted_CbGiG = np.maximum(predicted_CbGiG, 0)

    # Compute analytical probabilities for each gene degree
    analytical_probs = np.array([
        compute_analytical_edge_probability(
            deg_gene, deg_PW, n_edges_GpPW, n_genes, n_pathways
        )
        for deg_gene in all_gene_degrees
    ])

    # Composition: count × probability
    predicted_CbGiGpPW = np.sum(predicted_CbGiG * analytical_probs)
    predictions_analytical.append(predicted_CbGiGpPW)

predictions_analytical = np.array(predictions_analytical)
t_analytical = time.time() - t0

print(f"  Prediction time: {t_analytical:.2f}s")

# Variant 2: Empirical
print("\nVariant 2 - Computing predictions with empirical probabilities...")
t0 = time.time()

predictions_empirical = []
for i, (src, tgt) in enumerate(zip(sources_test, targets_test)):
    if (i + 1) % 1000 == 0:
        print(f"    Processed {i+1}/{len(sources_test)} pairs...")

    deg_C = CbG.getrow(src).nnz
    deg_PW = GpPW.getcol(tgt).nnz

    # Vectorized feature construction for CbGiG (same as v1)
    features_CbGiG = np.column_stack([
        np.full(n_deg, deg_C),
        all_gene_degrees,
        np.full(n_deg, np.log1p(deg_C)),
        np.log1p(all_gene_degrees),
        np.full(n_deg, deg_C) * all_gene_degrees,
        np.full(n_deg, np.log1p(deg_C)) * np.log1p(all_gene_degrees),
        np.full(n_deg, deg_C ** 2),
        all_gene_degrees ** 2
    ])

    # Predict CbGiG counts
    predicted_CbGiG = model_CbGiG.predict(features_CbGiG)
    predicted_CbGiG = np.maximum(predicted_CbGiG, 0)

    # Get empirical probabilities for each gene degree
    empirical_probs = np.array([
        empirical_probs_GpPW.get((int(deg_gene), deg_PW), 0.0)
        for deg_gene in all_gene_degrees
    ])

    # Composition: count × probability
    predicted_CbGiGpPW = np.sum(predicted_CbGiG * empirical_probs)
    predictions_empirical.append(predicted_CbGiGpPW)

predictions_empirical = np.array(predictions_empirical)
t_empirical = time.time() - t0

print(f"  Prediction time: {t_empirical:.2f}s")

print("\n" + "="*80)
print("RESULTS")
print("="*80)

# Variant 1 results
r_analytical = np.corrcoef(y_test, predictions_analytical)[0, 1]
r2_analytical = r2_score(y_test, predictions_analytical)
mae_analytical = mean_absolute_error(y_test, predictions_analytical)

print(f"\nVariant 1 (Analytical):")
print(f"  r = {r_analytical:.4f}")
print(f"  R² = {r2_analytical:.4f}")
print(f"  MAE = {mae_analytical:.4f}")
print(f"  Predicted range: [{predictions_analytical.min():.4f}, {predictions_analytical.max():.4f}]")
print(f"  Predicted mean: {predictions_analytical.mean():.4f}")

# Variant 2 results
r_empirical = np.corrcoef(y_test, predictions_empirical)[0, 1]
r2_empirical = r2_score(y_test, predictions_empirical)
mae_empirical = mean_absolute_error(y_test, predictions_empirical)

print(f"\nVariant 2 (Empirical):")
print(f"  r = {r_empirical:.4f}")
print(f"  R² = {r2_empirical:.4f}")
print(f"  MAE = {mae_empirical:.4f}")
print(f"  Predicted range: [{predictions_empirical.min():.4f}, {predictions_empirical.max():.4f}]")
print(f"  Predicted mean: {predictions_empirical.mean():.4f}")

print(f"\nActual:")
print(f"  Range: [{y_test.min():.4f}, {y_test.max():.4f}]")
print(f"  Mean: {y_test.mean():.4f}")

print("\n" + "="*80)
print("COMPARISON TO ALL EXPERIMENTS")
print("="*80)

print(f"\nExperiment 1 (aggregate degrees):        r = 0.32")
print(f"Experiment 2A (actual subpath counts):   r = 0.62")
print(f"Naive composition:                       r = 0.34")
print(f"Experiment 2B (count × count):           r = -0.09")
print(f"Experiment 2C-v1 (count × analytical):   r = {r_analytical:.4f}")
print(f"Experiment 2C-v2 (count × empirical):    r = {r_empirical:.4f}")

# Determine best variant
if r_analytical > r_empirical:
    best_variant = "Analytical"
    best_r = r_analytical
    best_predictions = predictions_analytical
else:
    best_variant = "Empirical"
    best_r = r_empirical
    best_predictions = predictions_empirical

print(f"\nBest variant: {best_variant} (r = {best_r:.4f})")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_list = []

results_list.append({
    'experiment': 'Experiment 2C-v1',
    'variant': 'Analytical',
    'metapath': 'CbGiGpPW',
    'description': 'Predicted counts × analytical edge probabilities',
    'n_test': len(y_test),
    'r_test': r_analytical,
    'r2_test': r2_analytical,
    'mae_test': mae_analytical,
    'time_predict': t_analytical,
    'success': r_analytical > 0.90
})

results_list.append({
    'experiment': 'Experiment 2C-v2',
    'variant': 'Empirical',
    'metapath': 'CbGiGpPW',
    'description': 'Predicted counts × empirical edge probabilities',
    'n_test': len(y_test),
    'r_test': r_empirical,
    'r2_test': r2_empirical,
    'mae_test': mae_empirical,
    'time_predict': t_empirical,
    'success': r_empirical > 0.90
})

df_results = pd.DataFrame(results_list)
df_results.to_csv(results_dir / 'experiment2c_results.csv', index=False)

# Create visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Analytical predictions
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, predictions_analytical, alpha=0.3, s=20)
ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
ax1.set_xlabel('True Path Count')
ax1.set_ylabel('Predicted Path Count')
ax1.set_title(f'2C-v1: Analytical (r={r_analytical:.3f})')
ax1.grid(alpha=0.3)

# Plot 2: Empirical predictions
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, predictions_empirical, alpha=0.3, s=20)
ax2.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
ax2.set_xlabel('True Path Count')
ax2.set_ylabel('Predicted Path Count')
ax2.set_title(f'2C-v2: Empirical (r={r_empirical:.3f})')
ax2.grid(alpha=0.3)

# Plot 3: Comparison
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(predictions_analytical, predictions_empirical, alpha=0.3, s=20)
min_val = min(predictions_analytical.min(), predictions_empirical.min())
max_val = max(predictions_analytical.max(), predictions_empirical.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax3.set_xlabel('Analytical Predictions')
ax3.set_ylabel('Empirical Predictions')
ax3.set_title('2C-v1 vs 2C-v2')
ax3.grid(alpha=0.3)

# Plot 4: All experiments comparison
ax4 = plt.subplot(2, 3, 4)
models = ['Exp 1', 'Exp 2A', 'Naive', 'Exp 2B', '2C-v1\n(Analytical)', '2C-v2\n(Empirical)']
correlations = [0.32, 0.62, 0.34, -0.09, r_analytical, r_empirical]
colors = ['red' if r < 0.7 else 'orange' if r < 0.9 else 'green' for r in correlations]
ax4.bar(models, correlations, color=colors, edgecolor='black', alpha=0.7)
ax4.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Success')
ax4.axhline(y=0.70, color='orange', linestyle='--', linewidth=2, label='Partial')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_ylabel('Test Correlation (r)')
ax4.set_title('All Experiments')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# Plot 5: Best variant residuals
ax5 = plt.subplot(2, 3, 5)
residuals = y_test - best_predictions
ax5.scatter(best_predictions, residuals, alpha=0.3, s=20)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Count')
ax5.set_ylabel('Residual')
ax5.set_title(f'Residuals ({best_variant})')
ax5.grid(alpha=0.3)

# Plot 6: Error distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax6.set_xlabel('Residual')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Error Distribution ({best_variant})')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'experiment2c_plots.png', dpi=150, bbox_inches='tight')

print(f"  Saved plots: {results_dir / 'experiment2c_plots.png'}")

print("\n" + "="*80)
print("EXPERIMENT 2C COMPLETE")
print("="*80)
print(f"\nBest Result: {best_variant} variant with r = {best_r:.4f}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2c_results.csv'}")
print(f"  {results_dir / 'experiment2c_plots.png'}")
