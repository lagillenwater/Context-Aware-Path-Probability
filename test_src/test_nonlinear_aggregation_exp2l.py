"""
Experiment 2L: Non-Linear Aggregation of 2-Hop Predictions

Test whether non-linear models can break the r=0.80 ceiling by learning
how to aggregate intermediate contributions.

Uses 2-hop model predictions (which work at r>0.95) as features,
then learns non-linear aggregation with Random Forest, Gradient Boosting, and Neural Net.

Training: Perm 1 (or mean of perms 1-K)
Validation: Mean of perms 11-20

Date: 2025-11-05
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
print("EXPERIMENT 2L: NON-LINEAR AGGREGATION OF 2-HOP PREDICTIONS")
print("Test non-linear models for breaking r=0.80 ceiling")
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
print("PHASE 1: Load Data and Train 2-Hop Models")
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

print("\nTraining 2-hop models on perm 1...")
CbG_1 = load_edge_matrix('CbG', 1)
GiG_1 = load_edge_matrix('GiG', 1)
GpPW_1 = load_edge_matrix('GpPW', 1)

CbGiG_1 = CbG_1 @ GiG_1
GiGpPW_1 = GiG_1 @ GpPW_1

# Train CbGiG model
print("  Training CbGiG model...")
CbGiG_1_coo = CbGiG_1.tocoo()
X_CbGiG = np.array([[compound_degrees[i], gene_degrees[j]]
                     for i, j in zip(CbGiG_1_coo.row, CbGiG_1_coo.col)])
y_CbGiG = np.array(CbGiG_1_coo.data, dtype=float)

model_CbGiG = LinearRegression()
model_CbGiG.fit(X_CbGiG, y_CbGiG)
print(f"    Trained on {len(X_CbGiG)} edges")

# Train GiGpPW model
print("  Training GiGpPW model...")
GiGpPW_1_coo = GiGpPW_1.tocoo()
X_GiGpPW = np.array([[gene_degrees[i], pathway_degrees[j]]
                      for i, j in zip(GiGpPW_1_coo.row, GiGpPW_1_coo.col)])
y_GiGpPW = np.array(GiGpPW_1_coo.data, dtype=float)

model_GiGpPW = LinearRegression()
model_GiGpPW.fit(X_GiGpPW, y_GiGpPW)
print(f"    Trained on {len(X_GiGpPW)} edges")


print("\n" + "="*80)
print("PHASE 2: Extract Aggregation Features")
print("="*80)

CbGiG_0_lil = CbGiG_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()

features_list = []

print("\nExtracting per-pair aggregation features...")
for idx, (C, PW) in enumerate(pairs):
    if idx % 2000 == 0:
        print(f"  Processed {idx}/{len(pairs)} pairs...")

    deg_C = compound_degrees[C]
    deg_PW = pathway_degrees[PW]

    # Identify intermediates from perm 0 topology
    genes_to_PW = GpPW_0_csr[:, PW].nonzero()[0]
    intermediates = [G2 for G2 in genes_to_PW if CbGiG_0_lil[C, G2] > 0]

    if len(intermediates) == 0:
        # No intermediates - all features are zero
        features = np.zeros(25)
        features[23] = deg_C  # Endpoint degrees still included
        features[24] = deg_PW
        features_list.append(features)
        continue

    # Compute 2-hop predictions for each intermediate
    pred_CbGiG_list = []
    pred_GiGpPW_list = []
    deg_intermediates = []
    products = []

    for G2 in intermediates:
        deg_G2 = gene_degrees[G2]
        deg_intermediates.append(deg_G2)

        # Predict 2-hop paths
        pred_CbGiG = model_CbGiG.predict([[deg_C, deg_G2]])[0]
        pred_GiGpPW = model_GiGpPW.predict([[deg_G2, deg_PW]])[0]

        pred_CbGiG_list.append(pred_CbGiG)
        pred_GiGpPW_list.append(pred_GiGpPW)
        products.append(pred_CbGiG * pred_GiGpPW)

    pred_CbGiG_arr = np.array(pred_CbGiG_list)
    pred_GiGpPW_arr = np.array(pred_GiGpPW_list)
    deg_intermediates_arr = np.array(deg_intermediates)
    products_arr = np.array(products)

    # Aggregate features (25 total)
    features = [
        # Basic aggregations
        np.sum(pred_CbGiG_arr),        # 0
        np.sum(pred_GiGpPW_arr),       # 1
        np.sum(products_arr),          # 2 - linear composition baseline

        # Count
        len(intermediates),            # 3

        # Degree statistics
        np.mean(deg_intermediates_arr),   # 4
        np.std(deg_intermediates_arr),    # 5
        np.min(deg_intermediates_arr),    # 6
        np.max(deg_intermediates_arr),    # 7
        np.median(deg_intermediates_arr), # 8

        # CbGiG prediction statistics
        np.mean(pred_CbGiG_arr),       # 9
        np.max(pred_CbGiG_arr),        # 10
        np.std(pred_CbGiG_arr),        # 11

        # GiGpPW prediction statistics
        np.mean(pred_GiGpPW_arr),      # 12
        np.max(pred_GiGpPW_arr),       # 13
        np.std(pred_GiGpPW_arr),       # 14

        # Product statistics
        np.mean(products_arr),         # 15
        np.max(products_arr),          # 16
        np.sum(products_arr**2),       # 17 - sum of squares

        # Additional aggregations
        np.min(pred_CbGiG_arr),        # 18
        np.min(pred_GiGpPW_arr),       # 19
        np.min(products_arr),          # 20

        # Ratios
        np.max(products_arr) / (np.sum(products_arr) + 1e-6),  # 21 - concentration
        np.std(products_arr),          # 22

        # Endpoint degrees
        deg_C,                         # 23
        deg_PW,                        # 24
    ]

    features_list.append(features)

X_features = np.array(features_list)
print(f"\nFeature matrix: {X_features.shape}")
print(f"  Feature 2 (linear composition) range: [{X_features[:, 2].min():.2f}, {X_features[:, 2].max():.2f}]")
print(f"  Feature 3 (n_intermediates) range: [{X_features[:, 3].min():.0f}, {X_features[:, 3].max():.0f}]")


print("\n" + "="*80)
print("PHASE 3: Compute Targets")
print("="*80)

print("\nComputing training target (perm 1)...")
CbGiGpPW_1 = CbGiG_1 @ GpPW_1
CbGiGpPW_1_lil = CbGiGpPW_1.tolil()
y_train_target = np.array([float(CbGiGpPW_1_lil[C, PW]) for C, PW in pairs])
print(f"  Mean: {y_train_target.mean():.3f}, Non-zero: {np.sum(y_train_target > 0)}")

print("\nComputing validation target (mean of perms 11-20)...")
val_counts = []
for perm in range(11, 21):
    print(f"  Loading perm {perm}...")
    CbG_p = load_edge_matrix('CbG', perm)
    GiG_p = load_edge_matrix('GiG', perm)
    GpPW_p = load_edge_matrix('GpPW', perm)
    CbGiG_p = CbG_p @ GiG_p
    CbGiGpPW_p = CbGiG_p @ GpPW_p
    CbGiGpPW_p_lil = CbGiGpPW_p.tolil()
    counts = np.array([float(CbGiGpPW_p_lil[C, PW]) for C, PW in pairs])
    val_counts.append(counts)

y_val_target = np.mean(val_counts, axis=0)
print(f"  Mean: {y_val_target.mean():.3f}, Non-zero: {np.sum(y_val_target > 0)}")

r_targets = pearsonr(y_train_target, y_val_target)[0]
print(f"\nTarget correlation: r = {r_targets:.4f}")


print("\n" + "="*80)
print("PHASE 4: Train Models")
print("="*80)

# Split data
X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
    X_features, y_train_target, y_val_target, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

results = {}

# Model 1: Linear baseline (composition only)
print("\n--- Model 1: Linear Baseline (composition sum only) ---")
model_linear_baseline = LinearRegression()
model_linear_baseline.fit(X_train[:, 2:3], y_train)
y_pred = model_linear_baseline.predict(X_test[:, 2:3])
r_val = pearsonr(y_pred, y_val_test)[0]
mae_val = mean_absolute_error(y_val_test, y_pred)
print(f"  r vs validation: {r_val:.4f}, MAE: {mae_val:.3f}")
results['Linear-Baseline'] = {'r': r_val, 'mae': mae_val, 'predictions': y_pred}

# Model 2: Linear with all features
print("\n--- Model 2: Linear Regression (all features) ---")
model_linear_all = LinearRegression()
model_linear_all.fit(X_train, y_train)
y_pred = model_linear_all.predict(X_test)
r_val = pearsonr(y_pred, y_val_test)[0]
mae_val = mean_absolute_error(y_val_test, y_pred)
print(f"  r vs validation: {r_val:.4f}, MAE: {mae_val:.3f}")
results['Linear-All'] = {'r': r_val, 'mae': mae_val, 'predictions': y_pred}

# Model 3: Random Forest
print("\n--- Model 3: Random Forest ---")
model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
r_val = pearsonr(y_pred, y_val_test)[0]
mae_val = mean_absolute_error(y_val_test, y_pred)
print(f"  r vs validation: {r_val:.4f}, MAE: {mae_val:.3f}")
results['RandomForest'] = {'r': r_val, 'mae': mae_val, 'predictions': y_pred}

# Get feature importance
feature_importance = model_rf.feature_importances_
print(f"\n  Top 5 features:")
top_idx = np.argsort(feature_importance)[-5:][::-1]
feature_names = ['sum_CbGiG', 'sum_GiGpPW', 'sum_products', 'n_intermediates',
                 'mean_deg', 'std_deg', 'min_deg', 'max_deg', 'median_deg',
                 'mean_pred_CbGiG', 'max_pred_CbGiG', 'std_pred_CbGiG',
                 'mean_pred_GiGpPW', 'max_pred_GiGpPW', 'std_pred_GiGpPW',
                 'mean_products', 'max_products', 'sum_products_sq',
                 'min_pred_CbGiG', 'min_pred_GiGpPW', 'min_products',
                 'concentration', 'std_products', 'deg_C', 'deg_PW']
for i in top_idx:
    print(f"    {feature_names[i]:20s}: {feature_importance[i]:.4f}")

# Model 4: Gradient Boosting
print("\n--- Model 4: Gradient Boosting ---")
model_gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_gb.fit(X_train, y_train)
y_pred = model_gb.predict(X_test)
r_val = pearsonr(y_pred, y_val_test)[0]
mae_val = mean_absolute_error(y_val_test, y_pred)
print(f"  r vs validation: {r_val:.4f}, MAE: {mae_val:.3f}")
results['GradientBoosting'] = {'r': r_val, 'mae': mae_val, 'predictions': y_pred}

# Model 5: Neural Network
print("\n--- Model 5: Neural Network ---")

class AggregationNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model_nn = AggregationNet(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)

# Train
for epoch in range(100):
    model_nn.train()
    optimizer.zero_grad()
    pred = model_nn(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Evaluate
model_nn.eval()
with torch.no_grad():
    y_pred = model_nn(X_test_t).numpy()

r_val = pearsonr(y_pred, y_val_test)[0]
mae_val = mean_absolute_error(y_val_test, y_pred)
print(f"  r vs validation: {r_val:.4f}, MAE: {mae_val:.3f}")
results['NeuralNet'] = {'r': r_val, 'mae': mae_val, 'predictions': y_pred}


print("\n" + "="*80)
print("PHASE 5: Summary")
print("="*80)

print("\nResults Summary (K=1, validate on mean perms 11-20):")
print(f"{'Model':<25s} {'r':>8s} {'MAE':>8s}")
print("-" * 45)
for name, res in results.items():
    print(f"{name:<25s} {res['r']:>8.4f} {res['mae']:>8.3f}")

best_model = max(results.items(), key=lambda x: x[1]['r'])
print(f"\nBest model: {best_model[0]} with r = {best_model[1]['r']:.4f}")

if best_model[1]['r'] > 0.95:
    print("\nSUCCESS: Achieved r > 0.95!")
elif best_model[1]['r'] > 0.85:
    print("\nPROMISING: Improvement over baseline (r=0.80), but not r>0.95")
    print("Consider testing with multiple permutations (K=2,3,4,5,10)")
else:
    print("\nFAILURE: No significant improvement over linear baseline")

# Save results
df_results = pd.DataFrame({
    'model': list(results.keys()),
    'r_validation': [res['r'] for res in results.values()],
    'mae_validation': [res['mae'] for res in results.values()]
})
df_results.to_csv(results_dir / 'experiment2l_results.csv', index=False)
print(f"\nSaved: {results_dir / 'experiment2l_results.csv'}")


print("\n" + "="*80)
print("EXPERIMENT 2L COMPLETE")
print("="*80)
