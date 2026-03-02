"""
Experiment 2L (Revised): Non-Linear Aggregation of 2-Hop Predictions

Improvements over original:
1. Use perm 0 consistently (topology, training, 2-hop models)
2. Validate against individual perms 11-20 (not averaged)
3. Report aggregated statistics (mean, std, min, max)

Training: Perm 0
Validation: Individual perms 11-20

Date: 2025-11-11
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2L (REVISED): NON-LINEAR AGGREGATION OF 2-HOP PREDICTIONS")
print("Improved design: consistent perm 0 usage, individual validation")
print("="*80)


def load_edge_matrix(edge_abbrev, perm_num):
    edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'
    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    return None


def sample_pairs_stratified(pathway_matrix, n_samples=10000, random_state=42):
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


print("\n" + "="*80)
print("PHASE 1: Load Data and Train 2-Hop Models (Perm 0)")
print("="*80)

print("\nLoading perm 0 topology...")
CbG_0 = load_edge_matrix('CbG', 0)
GiG_0 = load_edge_matrix('GiG', 0)
GpPW_0 = load_edge_matrix('GpPW', 0)

CbGiG_0 = CbG_0 @ GiG_0
CbGiGpPW_0 = CbGiG_0 @ GpPW_0

compound_degrees = np.array(CbG_0.sum(axis=1)).flatten()
gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()

print("\nSampling pairs...")
pairs = sample_pairs_stratified(CbGiGpPW_0, n_samples=10000, random_state=42)
print(f"  Sampled {len(pairs)} pairs")

print("\nTraining 2-hop models on perm 0...")
# Train CbGiG model
print("  Training CbGiG model...")
CbGiG_0_coo = CbGiG_0.tocoo()
X_CbGiG = np.array([[compound_degrees[i], gene_degrees[j]]
                     for i, j in zip(CbGiG_0_coo.row, CbGiG_0_coo.col)])
y_CbGiG = np.array(CbGiG_0_coo.data, dtype=float)

model_CbGiG = LinearRegression()
model_CbGiG.fit(X_CbGiG, y_CbGiG)
print(f"    Trained on {len(X_CbGiG)} edges")

# Train GiGpPW model
print("  Training GiGpPW model...")
GiGpPW_0 = GiG_0 @ GpPW_0
GiGpPW_0_coo = GiGpPW_0.tocoo()
X_GiGpPW = np.array([[gene_degrees[i], pathway_degrees[j]]
                      for i, j in zip(GiGpPW_0_coo.row, GiGpPW_0_coo.col)])
y_GiGpPW = np.array(GiGpPW_0_coo.data, dtype=float)

model_GiGpPW = LinearRegression()
model_GiGpPW.fit(X_GiGpPW, y_GiGpPW)
print(f"    Trained on {len(X_GiGpPW)} edges")


print("\n" + "="*80)
print("PHASE 2: Extract Aggregation Features")
print("="*80)

CbGiG_0_bool = CbGiG_0.astype(bool).tocsr()
GpPW_0_bool = GpPW_0.astype(bool).tocsr()

features_list = []

print("\nExtracting per-pair aggregation features...")
for idx, (C, PW) in enumerate(pairs):
    if idx % 2000 == 0:
        print(f"  Processed {idx}/{len(pairs)} pairs...")

    deg_C = compound_degrees[C]
    deg_PW = pathway_degrees[PW]

    # Identify intermediates using boolean masking (fastest method from benchmark)
    mask_C = CbGiG_0_bool[C, :].toarray().flatten()
    mask_PW = GpPW_0_bool[:, PW].toarray().flatten()
    intermediate_mask = mask_C & mask_PW
    intermediates = np.where(intermediate_mask)[0]

    if len(intermediates) == 0:
        features = np.zeros(25)
        features[23] = deg_C
        features[24] = deg_PW
        features_list.append(features)
        continue

    # Compute 2-hop predictions for each intermediate
    deg_intermediates = gene_degrees[intermediates]

    # Vectorized predictions
    X_CbGiG_batch = np.column_stack([np.repeat(deg_C, len(intermediates)), deg_intermediates])
    X_GiGpPW_batch = np.column_stack([deg_intermediates, np.repeat(deg_PW, len(intermediates))])

    pred_CbGiG_arr = model_CbGiG.predict(X_CbGiG_batch)
    pred_GiGpPW_arr = model_GiGpPW.predict(X_GiGpPW_batch)
    products_arr = pred_CbGiG_arr * pred_GiGpPW_arr

    # Aggregate features (25 total)
    features = [
        np.sum(pred_CbGiG_arr),        # 0
        np.sum(pred_GiGpPW_arr),       # 1
        np.sum(products_arr),          # 2
        len(intermediates),            # 3
        np.mean(deg_intermediates),    # 4
        np.std(deg_intermediates),     # 5
        np.min(deg_intermediates),     # 6
        np.max(deg_intermediates),     # 7
        np.median(deg_intermediates),  # 8
        np.mean(pred_CbGiG_arr),       # 9
        np.max(pred_CbGiG_arr),        # 10
        np.std(pred_CbGiG_arr),        # 11
        np.mean(pred_GiGpPW_arr),      # 12
        np.max(pred_GiGpPW_arr),       # 13
        np.std(pred_GiGpPW_arr),       # 14
        np.mean(products_arr),         # 15
        np.max(products_arr),          # 16
        np.sum(products_arr**2),       # 17
        np.min(pred_CbGiG_arr),        # 18
        np.min(pred_GiGpPW_arr),       # 19
        np.min(products_arr),          # 20
        np.max(products_arr) / (np.sum(products_arr) + 1e-6),  # 21
        np.std(products_arr),          # 22
        deg_C,                         # 23
        deg_PW,                        # 24
    ]

    features_list.append(features)

X_features = np.array(features_list)
print(f"\nFeature matrix: {X_features.shape}")


print("\n" + "="*80)
print("PHASE 3: Compute Training Target (Perm 0)")
print("="*80)

print("\nComputing training target (perm 0)...")
CbGiGpPW_0_lil = CbGiGpPW_0.tolil()
y_train_target = np.array([float(CbGiGpPW_0_lil[C, PW]) for C, PW in pairs])
print(f"  Mean: {y_train_target.mean():.3f}, Non-zero: {np.sum(y_train_target > 0)}")


print("\n" + "="*80)
print("PHASE 4: Train Models")
print("="*80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_train_target, test_size=0.2, random_state=42
)
# Store test pair indices for validation
test_indices = train_test_split(
    range(len(pairs)), test_size=0.2, random_state=42
)[1]
test_pairs = [pairs[i] for i in test_indices]

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

models = {}

# Model 1: Linear baseline (composition only)
print("\n--- Model 1: Linear Baseline (composition sum only) ---")
model_linear_baseline = LinearRegression()
model_linear_baseline.fit(X_train[:, 2:3], y_train)
models['Linear-Baseline'] = model_linear_baseline

# Model 2: Linear with all features
print("--- Model 2: Linear Regression (all features) ---")
model_linear_all = LinearRegression()
model_linear_all.fit(X_train, y_train)
models['Linear-All'] = model_linear_all

# Model 3: Random Forest
print("--- Model 3: Random Forest ---")
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)
models['RandomForest'] = model_rf

# Model 4: Gradient Boosting
print("--- Model 4: Gradient Boosting ---")
model_gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_gb.fit(X_train, y_train)
models['GradientBoosting'] = model_gb


print("\n" + "="*80)
print("PHASE 5: Individual Validation on Perms 11-20")
print("="*80)

# Get predictions once
predictions = {}
for name, model in models.items():
    if name == 'Linear-Baseline':
        predictions[name] = model.predict(X_test[:, 2:3])
    else:
        predictions[name] = model.predict(X_test)

# Validate against each permutation individually
print("\nValidating against individual permutations...")
validation_results = {name: {'r': [], 'mae': []} for name in models.keys()}

for perm in range(11, 21):
    print(f"\n  Loading perm {perm}...")
    CbG_p = load_edge_matrix('CbG', perm)
    GiG_p = load_edge_matrix('GiG', perm)
    GpPW_p = load_edge_matrix('GpPW', perm)
    CbGiG_p = CbG_p @ GiG_p
    CbGiGpPW_p = CbGiG_p @ GpPW_p
    CbGiGpPW_p_lil = CbGiGpPW_p.tolil()

    y_val_perm = np.array([float(CbGiGpPW_p_lil[C, PW]) for C, PW in test_pairs])

    for name, pred in predictions.items():
        r = pearsonr(pred, y_val_perm)[0]
        mae = mean_absolute_error(y_val_perm, pred)
        validation_results[name]['r'].append(r)
        validation_results[name]['mae'].append(mae)
        print(f"    {name:25s} r={r:.4f}, MAE={mae:.3f}")


print("\n" + "="*80)
print("PHASE 6: Aggregated Statistics")
print("="*80)

# Compute aggregated statistics
summary_stats = []
for name in models.keys():
    r_values = validation_results[name]['r']
    mae_values = validation_results[name]['mae']

    stats = {
        'model': name,
        'r_mean': np.mean(r_values),
        'r_std': np.std(r_values),
        'r_min': np.min(r_values),
        'r_max': np.max(r_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'mae_min': np.min(mae_values),
        'mae_max': np.max(mae_values),
    }
    summary_stats.append(stats)

df_summary = pd.DataFrame(summary_stats)
print("\nValidation Statistics (across perms 11-20):")
print(df_summary.to_string(index=False))

# Also evaluate on training permutation for comparison
print("\n" + "="*80)
print("Training Performance (Perm 0)")
print("="*80)

for name, pred in predictions.items():
    r_train = pearsonr(pred, y_test)[0]
    mae_train = mean_absolute_error(y_test, pred)
    print(f"{name:25s} r={r_train:.4f}, MAE={mae_train:.3f}")

# Feature importance from Random Forest
print("\n" + "="*80)
print("Feature Importance (Random Forest)")
print("="*80)

feature_importance = model_rf.feature_importances_
feature_names = ['sum_CbGiG', 'sum_GiGpPW', 'sum_products', 'n_intermediates',
                 'mean_deg', 'std_deg', 'min_deg', 'max_deg', 'median_deg',
                 'mean_pred_CbGiG', 'max_pred_CbGiG', 'std_pred_CbGiG',
                 'mean_pred_GiGpPW', 'max_pred_GiGpPW', 'std_pred_GiGpPW',
                 'mean_products', 'max_products', 'sum_products_sq',
                 'min_pred_CbGiG', 'min_pred_GiGpPW', 'min_products',
                 'concentration', 'std_products', 'deg_C', 'deg_PW']

print("\nTop 10 features:")
top_idx = np.argsort(feature_importance)[-10:][::-1]
for i in top_idx:
    print(f"  {feature_names[i]:20s}: {feature_importance[i]:.4f} ({100*feature_importance[i]:.1f}%)")

cumulative_importance = np.cumsum(sorted(feature_importance, reverse=True))
print(f"\nTop 5 features account for: {100*cumulative_importance[4]:.1f}% of importance")
print(f"Top 10 features account for: {100*cumulative_importance[9]:.1f}% of importance")

# Save results
print("\n" + "="*80)
print("Saving Results")
print("="*80)

# Save individual permutation results
individual_results = []
for perm_idx, perm in enumerate(range(11, 21)):
    for name in models.keys():
        individual_results.append({
            'permutation': perm,
            'model': name,
            'r': validation_results[name]['r'][perm_idx],
            'mae': validation_results[name]['mae'][perm_idx]
        })

df_individual = pd.DataFrame(individual_results)
df_individual.to_csv(results_dir / 'experiment2l_revised_individual.csv', index=False)
print(f"Saved: {results_dir / 'experiment2l_revised_individual.csv'}")

# Save summary statistics
df_summary.to_csv(results_dir / 'experiment2l_revised_summary.csv', index=False)
print(f"Saved: {results_dir / 'experiment2l_revised_summary.csv'}")

# Save feature importance
df_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
df_importance.to_csv(results_dir / 'experiment2l_revised_feature_importance.csv', index=False)
print(f"Saved: {results_dir / 'experiment2l_revised_feature_importance.csv'}")


print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

best_model = df_summary.loc[df_summary['r_mean'].idxmax()]
print(f"\nBest model: {best_model['model']}")
print(f"  Mean r: {best_model['r_mean']:.4f} ± {best_model['r_std']:.4f}")
print(f"  Range: [{best_model['r_min']:.4f}, {best_model['r_max']:.4f}]")
print(f"  Mean MAE: {best_model['mae_mean']:.3f} ± {best_model['mae_std']:.3f}")

if best_model['r_mean'] > 0.95:
    print("\nSUCCESS: Achieved r > 0.95!")
elif best_model['r_mean'] > 0.85:
    print("\nPROMISING: r > 0.85, close to target")
    if best_model['r_std'] < 0.05:
        print("  Consistent across permutations (std < 0.05)")
    else:
        print("  Variable across permutations (std >= 0.05)")
else:
    print("\nFAILURE: r < 0.85")

print("\n" + "="*80)
print("EXPERIMENT 2L (REVISED) COMPLETE")
print("="*80)
