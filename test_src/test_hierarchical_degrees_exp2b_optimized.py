"""
Hierarchical Path Prediction - Experiment 2B (Optimized)

Degree-Stratified Compositional Prediction using vectorized operations.

Approach:
1. Train models for CbGiG and GiGpPW using degrees (like yesterday)
2. For each test pair, vectorize predictions across all gene degrees
3. Compute: predicted_CbGiGpPW = sum(pred_CbGiG[d] * pred_GiGpPW[d] for all d)

This tests if compositional multiplication works when stratified by degree.

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
print("HIERARCHICAL PATH PREDICTION - EXPERIMENT 2B (OPTIMIZED)")
print("Degree-Stratified Compositional Prediction (Vectorized)")
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


print("\n" + "="*80)
print("STEP 1: Train Subpath Models")
print("="*80)

print("\nLoading edge matrices...")
CbG = load_edge_matrix('CbG')
GiG = load_edge_matrix('GiG')
GpPW = load_edge_matrix('GpPW')

print(f"  CbG: {CbG.shape}, {CbG.nnz:,} edges")
print(f"  GiG: {GiG.shape}, {GiG.nnz:,} edges")
print(f"  GpPW: {GpPW.shape}, {GpPW.nnz:,} edges")

print("\nComputing subpath counts (ground truth)...")
t0 = time.time()
CbGiG = CbG @ GiG
GiGpPW = GiG @ GpPW
CbGiGpPW = CbGiG @ GpPW
t_compute = time.time() - t0

print(f"  CbGiG: {CbGiG.shape}, {CbGiG.nnz:,} non-zero")
print(f"  GiGpPW: {GiGpPW.shape}, {GiGpPW.nnz:,} non-zero")
print(f"  CbGiGpPW: {CbGiGpPW.shape}, {CbGiGpPW.nnz:,} non-zero")
print(f"  Computation time: {t_compute:.3f}s")

# Get gene degree distribution
gene_degrees = np.array(GiG.sum(axis=1)).flatten()
max_gene_degree = int(gene_degrees.max())
print(f"\n  Gene degrees: min={gene_degrees.min()}, max={max_gene_degree}, "
      f"median={np.median(gene_degrees):.0f}")

# Train Model 1: CbGiG from degrees
print("\n" + "-"*80)
print("Training Model 1: CbGiG from (deg_C, deg_G2)")
print("-"*80)

print("\nSampling pairs for CbGiG...")
src_1, tgt_1, y_1 = sample_node_pairs_for_edge(
    CbGiG, n_samples=10000, target_ratio=0.5, random_state=42
)

print(f"  Sampled {len(y_1)} pairs")
print(f"  Non-zero: {np.sum(y_1 > 0)}")

print("\nExtracting degree features...")
X_1_list = []
for src, tgt in zip(src_1, tgt_1):
    features = extract_degree_features_for_pair(src, tgt, CbG, GiG)
    X_1_list.append(features)
X_1 = np.array(X_1_list)

print(f"  Feature matrix: {X_1.shape}")

print("\nTraining linear regression...")
t0 = time.time()
model_CbGiG = LinearRegression()
model_CbGiG.fit(X_1, y_1)
t_train_1 = time.time() - t0

y_1_pred = model_CbGiG.predict(X_1)
r_1 = np.corrcoef(y_1, y_1_pred)[0, 1]

print(f"  Training time: {t_train_1:.4f}s")
print(f"  Training r: {r_1:.4f}")

# Train Model 2: GiGpPW from degrees
print("\n" + "-"*80)
print("Training Model 2: GiGpPW from (deg_G1, deg_PW)")
print("-"*80)

print("\nSampling pairs for GiGpPW...")
src_2, tgt_2, y_2 = sample_node_pairs_for_edge(
    GiGpPW, n_samples=10000, target_ratio=0.5, random_state=43
)

print(f"  Sampled {len(y_2)} pairs")
print(f"  Non-zero: {np.sum(y_2 > 0)}")

print("\nExtracting degree features...")
X_2_list = []
for src, tgt in zip(src_2, tgt_2):
    features = extract_degree_features_for_pair(src, tgt, GiG, GpPW)
    X_2_list.append(features)
X_2 = np.array(X_2_list)

print(f"  Feature matrix: {X_2.shape}")

print("\nTraining linear regression...")
t0 = time.time()
model_GiGpPW = LinearRegression()
model_GiGpPW.fit(X_2, y_2)
t_train_2 = time.time() - t0

y_2_pred = model_GiGpPW.predict(X_2)
r_2 = np.corrcoef(y_2, y_2_pred)[0, 1]

print(f"  Training time: {t_train_2:.4f}s")
print(f"  Training r: {r_2:.4f}")

print("\n" + "="*80)
print("STEP 2: Vectorized Degree-Stratified Prediction")
print("="*80)

print("\nSampling pairs for CbGiGpPW...")
sources_test, targets_test, y_test = sample_node_pairs_for_edge(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=45
)

print(f"  Test: {len(y_test)} pairs, {np.sum(y_test > 0)} with paths")
print(f"  Test range: [{y_test.min():.1f}, {y_test.max():.1f}]")

# Prepare all gene degrees for vectorization
all_gene_degrees = np.arange(1, max_gene_degree + 1)
n_deg = len(all_gene_degrees)

print(f"\n  Will iterate over {n_deg} gene degree values")
print(f"  Using vectorized prediction (batch size = {n_deg})")

print("\nComputing predictions...")
t0 = time.time()

predictions = []
for i, (src, tgt) in enumerate(zip(sources_test, targets_test)):
    if (i + 1) % 1000 == 0:
        print(f"    Processed {i+1}/{len(sources_test)} pairs...")

    deg_C = CbG.getrow(src).nnz
    deg_PW = GpPW.getcol(tgt).nnz

    # Vectorized feature construction for CbGiG
    features_CbGiG = np.column_stack([
        np.full(n_deg, deg_C),                           # deg_C repeated
        all_gene_degrees,                                 # deg_G varies
        np.full(n_deg, np.log1p(deg_C)),                 # log(deg_C)
        np.log1p(all_gene_degrees),                       # log(deg_G)
        np.full(n_deg, deg_C) * all_gene_degrees,        # deg_C × deg_G
        np.full(n_deg, np.log1p(deg_C)) * np.log1p(all_gene_degrees),
        np.full(n_deg, deg_C ** 2),                      # deg_C²
        all_gene_degrees ** 2                             # deg_G²
    ])

    # Vectorized feature construction for GiGpPW
    features_GiGpPW = np.column_stack([
        all_gene_degrees,                                 # deg_G varies
        np.full(n_deg, deg_PW),                          # deg_PW repeated
        np.log1p(all_gene_degrees),                       # log(deg_G)
        np.full(n_deg, np.log1p(deg_PW)),                # log(deg_PW)
        all_gene_degrees * np.full(n_deg, deg_PW),       # deg_G × deg_PW
        np.log1p(all_gene_degrees) * np.full(n_deg, np.log1p(deg_PW)),
        all_gene_degrees ** 2,                            # deg_G²
        np.full(n_deg, deg_PW ** 2)                      # deg_PW²
    ])

    # Batch predictions
    predicted_CbGiG = model_CbGiG.predict(features_CbGiG)
    predicted_GiGpPW = model_GiGpPW.predict(features_GiGpPW)

    # Clip negative predictions
    predicted_CbGiG = np.maximum(predicted_CbGiG, 0)
    predicted_GiGpPW = np.maximum(predicted_GiGpPW, 0)

    # Degree-stratified composition
    predicted_CbGiGpPW = np.sum(predicted_CbGiG * predicted_GiGpPW)

    predictions.append(predicted_CbGiGpPW)

predictions = np.array(predictions)
t_predict = time.time() - t0

print(f"\n  Prediction time: {t_predict:.2f}s")
print(f"  Time per pair: {t_predict/len(sources_test)*1000:.1f}ms")

print("\n" + "="*80)
print("RESULTS")
print("="*80)

r_test = np.corrcoef(y_test, predictions)[0, 1]
r2_test = r2_score(y_test, predictions)
mae_test = mean_absolute_error(y_test, predictions)

print(f"\nTest Performance:")
print(f"  r = {r_test:.4f}")
print(f"  R² = {r2_test:.4f}")
print(f"  MAE = {mae_test:.2f}")

print(f"\nPrediction Statistics:")
print(f"  Min: {predictions.min():.2f}")
print(f"  Max: {predictions.max():.2f}")
print(f"  Mean: {predictions.mean():.2f}")
print(f"  Median: {np.median(predictions):.2f}")

print("\n" + "="*80)
print("COMPARISON TO BASELINES")
print("="*80)

print(f"\nExperiment 1 (aggregate degrees):      r = 0.32")
print(f"Experiment 2A (actual subpath counts): r = 0.62")
print(f"Naive composition:                     r = 0.34")
print(f"Experiment 2B (degree-stratified):    r = {r_test:.4f}")

if r_test > 0.62:
    improvement = ((r_test - 0.62) / 0.62) * 100
    print(f"\nImprovement over Exp 2A: {improvement:.1f}%")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if r_test > 0.90:
    status = "SUCCESS"
    interpretation = (
        f"Degree-stratified composition WORKS with r = {r_test:.4f} > 0.90. "
        "Compositional multiplication holds when stratified by intermediate degrees. "
        "This enables efficient prediction of longer paths."
    )
elif r_test > 0.70:
    status = "PARTIAL SUCCESS"
    interpretation = (
        f"Degree-stratified composition shows promise with r = {r_test:.4f}. "
        "Captures significant structure but may need refinement for production use."
    )
else:
    status = "FAILURE"
    interpretation = (
        f"Degree-stratified composition does not work well with r = {r_test:.4f} < 0.70. "
        "Compositional assumption may not hold even when stratified by degree."
    )

print(f"\nResult: {status}")
print(f"\n{interpretation}")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'experiment': 'Experiment 2B (Optimized)',
    'metapath': 'CbGiGpPW',
    'description': 'Degree-stratified compositional prediction (vectorized)',
    'n_test': len(y_test),
    'r_test': r_test,
    'r2_test': r2_test,
    'mae_test': mae_test,
    'r_subpath_model1': r_1,
    'r_subpath_model2': r_2,
    'time_train_subpaths': t_train_1 + t_train_2,
    'time_predict': t_predict,
    'status': status,
    'success': r_test > 0.90
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2b_optimized_results.csv', index=False)

# Create visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Predicted vs actual
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, predictions, alpha=0.3, s=20)
ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
ax1.set_xlabel('True Path Count')
ax1.set_ylabel('Predicted Path Count')
ax1.set_title(f'Test Set (r={r_test:.3f})')
ax1.grid(alpha=0.3)

# Plot 2: Log scale
ax2 = plt.subplot(2, 3, 2)
mask = (y_test > 0) & (predictions > 0)
ax2.scatter(np.log1p(y_test[mask]), np.log1p(predictions[mask]), alpha=0.3, s=20)
max_val = max(np.log1p(y_test[mask]).max(), np.log1p(predictions[mask]).max())
ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax2.set_xlabel('log(True Count + 1)')
ax2.set_ylabel('log(Predicted Count + 1)')
ax2.set_title('Log Scale (non-zero only)')
ax2.grid(alpha=0.3)

# Plot 3: Residuals
ax3 = plt.subplot(2, 3, 3)
residuals = y_test - predictions
ax3.scatter(predictions, residuals, alpha=0.3, s=20)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Count')
ax3.set_ylabel('Residual (True - Predicted)')
ax3.set_title('Residuals')
ax3.grid(alpha=0.3)

# Plot 4: Model comparison
ax4 = plt.subplot(2, 3, 4)
models = ['Exp 1\n(Agg Deg)', 'Exp 2A\n(Actual)', 'Naive\nComp', 'Exp 2B\n(Stratified)']
correlations = [0.32, 0.62, 0.34, r_test]
colors = ['red' if r < 0.7 else 'orange' if r < 0.9 else 'green' for r in correlations]
ax4.bar(models, correlations, color=colors, edgecolor='black', alpha=0.7)
ax4.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Success')
ax4.axhline(y=0.70, color='orange', linestyle='--', linewidth=2, label='Partial')
ax4.set_ylabel('Test Correlation (r)')
ax4.set_title('Model Comparison')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# Plot 5: Error distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Residual')
ax5.set_ylabel('Frequency')
ax5.set_title(f'Error Distribution (MAE={mae_test:.2f})')
ax5.grid(alpha=0.3)

# Plot 6: Stratified by degree
ax6 = plt.subplot(2, 3, 6)
deg_C_test = np.array([CbG.getrow(s).nnz for s in sources_test])
deg_PW_test = np.array([GpPW.getcol(t).nnz for t in targets_test])

# Bin by degree quartiles
deg_C_bins = np.percentile(deg_C_test, [0, 50, 100])
deg_PW_bins = np.percentile(deg_PW_test, [0, 50, 100])

labels = []
corrs = []
for i in range(2):
    for j in range(2):
        mask = ((deg_C_test >= deg_C_bins[i]) & (deg_C_test < deg_C_bins[i+1]) &
                (deg_PW_test >= deg_PW_bins[j]) & (deg_PW_test < deg_PW_bins[j+1]))
        if mask.sum() > 10:
            r_subset = np.corrcoef(y_test[mask], predictions[mask])[0, 1]
            labels.append(f"{'Low' if i==0 else 'High'}C\n{'Low' if j==0 else 'High'}PW")
            corrs.append(r_subset)

ax6.bar(labels, corrs, edgecolor='black', alpha=0.7)
ax6.axhline(y=r_test, color='r', linestyle='--', linewidth=2, label='Overall')
ax6.set_ylabel('Correlation (r)')
ax6.set_title('Stratified by Endpoint Degrees')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_dir / 'experiment2b_optimized_plots.png', dpi=150, bbox_inches='tight')

print(f"  Saved plots: {results_dir / 'experiment2b_optimized_plots.png'}")

print("\n" + "="*80)
print("EXPERIMENT 2B (OPTIMIZED) COMPLETE")
print("="*80)
print(f"\nResult: {status}")
print(f"Test correlation: r = {r_test:.4f}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2b_optimized_results.csv'}")
print(f"  {results_dir / 'experiment2b_optimized_plots.png'}")
