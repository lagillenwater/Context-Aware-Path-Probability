"""
Hierarchical Path Prediction - Experiment 2B

True hierarchical approach combining:
1. Hierarchical degree features (like yesterday's success)
2. Predicted pathway counts from trained subpath models
3. Interactions between degrees and predictions

For CbGiGpPW:
- Train models for CbGiG and GiGpPW using degrees
- Use these models to predict subpath counts
- Combine degree features + predicted counts to predict full path

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
print("HIERARCHICAL PATH PREDICTION - EXPERIMENT 2B")
print("True Hierarchical: Degrees + Predicted Counts")
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
    """
    Extract degree-based features for a single pair.
    Like yesterday's successful approach.
    """
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
print(f"  Range: [{y_1.min():.1f}, {y_1.max():.1f}]")

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
print(f"  Range: [{y_2.min():.1f}, {y_2.max():.1f}]")

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
print("STEP 2: Generate Features for Full Path")
print("="*80)

print("\nSampling pairs for CbGiGpPW...")
sources_train, targets_train, y_train = sample_node_pairs_for_edge(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=44
)
sources_test, targets_test, y_test = sample_node_pairs_for_edge(
    CbGiGpPW, n_samples=5000, target_ratio=0.5, random_state=45
)

print(f"  Train: {len(y_train)} pairs, {np.sum(y_train > 0)} with paths")
print(f"  Test: {len(y_test)} pairs, {np.sum(y_test > 0)} with paths")

print("\nBuilding hierarchical features...")
print("  For each (Compound, Pathway) pair:")
print("    - Extract endpoint degrees")
print("    - Extract intermediate gene degrees")
print("    - Use trained models to predict subpath counts")
print("    - Combine into rich feature set")

CbG_csr = CbG.tocsr()
GiG_csr = GiG.tocsr()
GpPW_csr = GpPW.tocsr()
CbGiG_csr = CbGiG.tocsr()
GiGpPW_csr = GiGpPW.tocsr()


def extract_hierarchical_features(src, tgt):
    """
    Extract full hierarchical feature set combining:
    - Degree features at each level
    - Predicted pathway counts from trained models
    - Cross-feature interactions
    """
    # Group 1: Endpoint degrees
    deg_C = CbG_csr.getrow(src).nnz
    deg_PW = GpPW_csr.getcol(tgt).nnz

    # Group 2: Intermediate gene degrees
    genes_from_C = CbG_csr.getrow(src).nonzero()[1]
    genes_to_PW = GpPW_csr.getcol(tgt).nonzero()[0]

    if len(genes_from_C) > 0:
        degs_G1 = [GiG_csr.getrow(g).nnz for g in genes_from_C]
        avg_deg_G1 = np.mean(degs_G1)
        max_deg_G1 = np.max(degs_G1)
    else:
        avg_deg_G1 = 0
        max_deg_G1 = 0

    if len(genes_to_PW) > 0:
        degs_G2 = [GiG_csr.getcol(g).nnz for g in genes_to_PW]
        avg_deg_G2 = np.mean(degs_G2)
        max_deg_G2 = np.max(degs_G2)
    else:
        avg_deg_G2 = 0
        max_deg_G2 = 0

    # Group 3: Predicted pathway counts
    # For CbGiG: predict C -> all G2
    CbGiG_row = CbGiG_csr.getrow(src).toarray().flatten()
    nonzero_G2 = np.nonzero(CbGiG_row)[0]

    if len(nonzero_G2) > 0:
        # Build features for prediction
        pred_features_CbGiG = []
        for g2 in nonzero_G2:
            feat = extract_degree_features_for_pair(src, g2, CbG_csr, GiG_csr)
            pred_features_CbGiG.append(feat)
        pred_features_CbGiG = np.array(pred_features_CbGiG)

        # Predict
        predicted_CbGiG_counts = model_CbGiG.predict(pred_features_CbGiG)
        predicted_CbGiG_counts = np.maximum(predicted_CbGiG_counts, 0)

        total_pred_CbGiG = np.sum(predicted_CbGiG_counts)
        max_pred_CbGiG = np.max(predicted_CbGiG_counts)
        avg_pred_CbGiG = np.mean(predicted_CbGiG_counts)
    else:
        total_pred_CbGiG = 0
        max_pred_CbGiG = 0
        avg_pred_CbGiG = 0

    # For GiGpPW: predict all G1 -> PW
    GiGpPW_col = GiGpPW_csr.getcol(tgt).toarray().flatten()
    nonzero_G1 = np.nonzero(GiGpPW_col)[0]

    if len(nonzero_G1) > 0:
        pred_features_GiGpPW = []
        for g1 in nonzero_G1:
            feat = extract_degree_features_for_pair(g1, tgt, GiG_csr, GpPW_csr)
            pred_features_GiGpPW.append(feat)
        pred_features_GiGpPW = np.array(pred_features_GiGpPW)

        predicted_GiGpPW_counts = model_GiGpPW.predict(pred_features_GiGpPW)
        predicted_GiGpPW_counts = np.maximum(predicted_GiGpPW_counts, 0)

        total_pred_GiGpPW = np.sum(predicted_GiGpPW_counts)
        max_pred_GiGpPW = np.max(predicted_GiGpPW_counts)
        avg_pred_GiGpPW = np.mean(predicted_GiGpPW_counts)
    else:
        total_pred_GiGpPW = 0
        max_pred_GiGpPW = 0
        avg_pred_GiGpPW = 0

    # Combine into feature vector
    features = [
        # Endpoint degrees (raw and log)
        deg_C,
        deg_PW,
        np.log1p(deg_C),
        np.log1p(deg_PW),

        # Intermediate degrees
        avg_deg_G1,
        max_deg_G1,
        avg_deg_G2,
        max_deg_G2,
        np.log1p(avg_deg_G1),
        np.log1p(avg_deg_G2),

        # Predicted counts
        total_pred_CbGiG,
        max_pred_CbGiG,
        avg_pred_CbGiG,
        total_pred_GiGpPW,
        max_pred_GiGpPW,
        avg_pred_GiGpPW,
        np.log1p(total_pred_CbGiG),
        np.log1p(total_pred_GiGpPW),

        # Degree interactions
        deg_C * deg_PW,
        deg_C * avg_deg_G1,
        avg_deg_G2 * deg_PW,
        deg_C * avg_deg_G1 * avg_deg_G2 * deg_PW,

        # Degree-prediction interactions
        deg_C * total_pred_CbGiG,
        deg_PW * total_pred_GiGpPW,
        total_pred_CbGiG * total_pred_GiGpPW,

        # Polynomials
        deg_C ** 2,
        deg_PW ** 2
    ]

    return features


feature_names = [
    'deg_C', 'deg_PW', 'log_deg_C', 'log_deg_PW',
    'avg_deg_G1', 'max_deg_G1', 'avg_deg_G2', 'max_deg_G2',
    'log_avg_deg_G1', 'log_avg_deg_G2',
    'total_pred_CbGiG', 'max_pred_CbGiG', 'avg_pred_CbGiG',
    'total_pred_GiGpPW', 'max_pred_GiGpPW', 'avg_pred_GiGpPW',
    'log_total_pred_CbGiG', 'log_total_pred_GiGpPW',
    'deg_C×deg_PW', 'deg_C×avg_deg_G1', 'avg_deg_G2×deg_PW',
    'deg_C×avg_deg_G1×avg_deg_G2×deg_PW',
    'deg_C×total_pred_CbGiG', 'deg_PW×total_pred_GiGpPW',
    'total_pred_CbGiG×total_pred_GiGpPW',
    'deg_C²', 'deg_PW²'
]

print("\n  Processing training set...")
X_train_list = []
for src, tgt in zip(sources_train, targets_train):
    features = extract_hierarchical_features(src, tgt)
    X_train_list.append(features)
X_train = np.array(X_train_list)

print("\n  Processing test set...")
X_test_list = []
for src, tgt in zip(sources_test, targets_test):
    features = extract_hierarchical_features(src, tgt)
    X_test_list.append(features)
X_test = np.array(X_test_list)

print(f"\n  Feature matrix: {X_train.shape}")
print(f"  Features: {len(feature_names)}")

print("\n" + "="*80)
print("STEP 3: Train Composition Model")
print("="*80)

print("\nTraining full model (degrees + predicted counts)...")
t0 = time.time()
model_full = LinearRegression()
model_full.fit(X_train, y_train)
t_train_full = time.time() - t0

y_train_pred = model_full.predict(X_train)
y_test_pred = model_full.predict(X_test)

r_train = np.corrcoef(y_train, y_train_pred)[0, 1]
r_test = np.corrcoef(y_test, y_test_pred)[0, 1]
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"  Training time: {t_train_full:.4f}s")

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nFull Model Performance:")
print(f"  Train: r = {r_train:.4f}, R² = {r2_train:.4f}, MAE = {mae_train:.2f}")
print(f"  Test:  r = {r_test:.4f}, R² = {r2_test:.4f}, MAE = {mae_test:.2f}")

# Baseline comparisons
print("\n" + "="*80)
print("BASELINE COMPARISONS")
print("="*80)

# Baseline A: Degrees only (endpoints + intermediate averages)
deg_idx = [0, 1, 2, 3, 4, 6, 8, 9, 18, 19, 20, 25, 26]
X_train_deg = X_train[:, deg_idx]
X_test_deg = X_test[:, deg_idx]

model_deg = LinearRegression()
model_deg.fit(X_train_deg, y_train)
y_test_deg = model_deg.predict(X_test_deg)
r_deg = np.corrcoef(y_test, y_test_deg)[0, 1]

# Baseline B: Predicted counts only
pred_idx = [10, 11, 12, 13, 14, 15, 16, 17, 24]
X_train_pred = X_train[:, pred_idx]
X_test_pred = X_test[:, pred_idx]

model_pred = LinearRegression()
model_pred.fit(X_train_pred, y_train)
y_test_pred_only = model_pred.predict(X_test_pred)
r_pred = np.corrcoef(y_test, y_test_pred_only)[0, 1]

print(f"\nBaseline A (Degrees only):          r = {r_deg:.4f}")
print(f"Baseline B (Predicted counts only): r = {r_pred:.4f}")
print(f"Full Model (Degrees + Predictions): r = {r_test:.4f}")

print(f"\nTop 10 Feature Coefficients (by absolute value):")
abs_coefs = np.abs(model_full.coef_)
top_idx = np.argsort(abs_coefs)[::-1][:10]
for idx in top_idx:
    print(f"  {feature_names[idx]:35s}: {model_full.coef_[idx]:12.6f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if r_test > 0.90:
    status = "SUCCESS"
    interpretation = (
        f"Hierarchical composition WORKS with r = {r_test:.4f} > 0.90. "
        "Combining degree features with predicted subpath counts enables "
        "accurate prediction of longer paths."
    )
elif r_test > 0.70:
    status = "PARTIAL SUCCESS"
    interpretation = (
        f"Hierarchical composition shows promise with r = {r_test:.4f}. "
        "The approach captures significant structure but may benefit from "
        "non-linear models or additional features."
    )
else:
    status = "FAILURE"
    interpretation = (
        f"Hierarchical composition does not work well with r = {r_test:.4f} < 0.70. "
        "Even combining degrees and predicted counts, length-3 paths may require "
        "alternative approaches."
    )

print(f"\nResult: {status}")
print(f"\n{interpretation}")

print(f"\nComparison to Previous Experiments:")
print(f"  Experiment 1 (aggregate degrees):      r = 0.32")
print(f"  Experiment 2A (actual subpath counts): r = 0.62")
print(f"  Experiment 2B (degrees + predictions): r = {r_test:.4f}")

if r_test > 0.62:
    improvement = ((r_test - 0.62) / 0.62) * 100
    print(f"  Improvement over 2A: {improvement:.1f}%")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'experiment': 'Experiment 2B',
    'metapath': 'CbGiGpPW',
    'description': 'Hierarchical degrees + predicted counts',
    'n_train': len(y_train),
    'n_test': len(y_test),
    'r_train': r_train,
    'r_test': r_test,
    'r2_train': r2_train,
    'r2_test': r2_test,
    'mae_train': mae_train,
    'mae_test': mae_test,
    'r_subpath_model1': r_1,
    'r_subpath_model2': r_2,
    'r_baseline_degrees': r_deg,
    'r_baseline_predictions': r_pred,
    'time_train_subpaths': t_train_1 + t_train_2,
    'time_train_composition': t_train_full,
    'status': status,
    'success': r_test > 0.90
}

df_results = pd.DataFrame([results])
df_results.to_csv(results_dir / 'experiment2b_results.csv', index=False)

df_features = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model_full.coef_,
    'abs_coefficient': np.abs(model_full.coef_)
})
df_features = df_features.sort_values('abs_coefficient', ascending=False)
df_features.to_csv(results_dir / 'experiment2b_features.csv', index=False)

# Create visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Train predictions
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_train, y_train_pred, alpha=0.3, s=20)
ax1.plot([0, y_train.max()], [0, y_train.max()], 'r--', linewidth=2)
ax1.set_xlabel('True Path Count')
ax1.set_ylabel('Predicted Path Count')
ax1.set_title(f'Train Set (r={r_train:.3f})')
ax1.grid(alpha=0.3)

# Plot 2: Test predictions
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, y_test_pred, alpha=0.3, s=20)
ax2.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
ax2.set_xlabel('True Path Count')
ax2.set_ylabel('Predicted Path Count')
ax2.set_title(f'Test Set (r={r_test:.3f})')
ax2.grid(alpha=0.3)

# Plot 3: Feature importance
ax3 = plt.subplot(2, 3, 3)
top_10_idx = np.argsort(np.abs(model_full.coef_))[::-1][:10]
y_pos = np.arange(10)
ax3.barh(y_pos, np.abs(model_full.coef_)[top_10_idx], edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels([feature_names[i] for i in top_10_idx], fontsize=8)
ax3.set_xlabel('|Coefficient|')
ax3.set_title('Top 10 Features')
ax3.grid(alpha=0.3, axis='x')

# Plot 4: Model comparison
ax4 = plt.subplot(2, 3, 4)
models = ['Exp 1\n(Degrees)', 'Exp 2A\n(Actual)', 'Degrees\nOnly',
          'Preds\nOnly', 'Exp 2B\n(Full)']
correlations = [0.32, 0.62, r_deg, r_pred, r_test]
colors = ['red' if r < 0.7 else 'orange' if r < 0.9 else 'green'
          for r in correlations]
ax4.bar(models, correlations, color=colors, edgecolor='black', alpha=0.7)
ax4.axhline(y=0.90, color='green', linestyle='--', linewidth=2)
ax4.axhline(y=0.70, color='orange', linestyle='--', linewidth=2)
ax4.set_ylabel('Test Correlation (r)')
ax4.set_title('Model Comparison')
ax4.grid(alpha=0.3, axis='y')

# Plot 5: Residuals
ax5 = plt.subplot(2, 3, 5)
residuals_test = y_test - y_test_pred
ax5.scatter(y_test_pred, residuals_test, alpha=0.3, s=20)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Path Count')
ax5.set_ylabel('Residual')
ax5.set_title('Test Residuals')
ax5.grid(alpha=0.3)

# Plot 6: Error distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax6.set_xlabel('Residual')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Error Distribution (MAE={mae_test:.2f})')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'experiment2b_plots.png', dpi=150, bbox_inches='tight')

print(f"\n  Saved plots: {results_dir / 'experiment2b_plots.png'}")

print("\n" + "="*80)
print("EXPERIMENT 2B COMPLETE")
print("="*80)
print(f"\nResult: {status}")
print(f"Test correlation: r = {r_test:.4f}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2b_results.csv'}")
print(f"  {results_dir / 'experiment2b_features.csv'}")
print(f"  {results_dir / 'experiment2b_plots.png'}")
