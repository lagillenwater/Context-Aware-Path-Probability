"""
Experiment 2F: Degree-Stratified Composition for Null Models

Goal: Predict E[CbGiGpPW | deg_C, deg_PW] by composing degree-based models
      for null distribution estimation (anomaly detection use case)

Approach:
1. Train GiGpPW count model (predict counts, not probabilities)
2. Decompose actual pathways by G2 degree
3. Compose: sum over G2 degrees with estimated n_genes
4. Diagnose: does G1 degree distribution matter?

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
from collections import defaultdict
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2F: DEGREE-STRATIFIED COMPOSITION FOR NULL MODELS")
print("Stratify by G2 degree, check if G1 matters")
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
    """Extract 5 degree features."""
    return np.array([
        deg_u, deg_v, deg_u * deg_v, deg_u ** 2, deg_v ** 2
    ], dtype=np.float64)


print("\n" + "="*80)
print("PHASE 1: Train CbGiG and GiGpPW Count Models")
print("="*80)

print("\nLoading perm 0 edges...")
CbG_0 = load_edge_matrix('CbG', perm_num=0)
GiG_0 = load_edge_matrix('GiG', perm_num=0)
GpPW_0 = load_edge_matrix('GpPW', perm_num=0)

print(f"  CbG: {CbG_0.shape}, {CbG_0.nnz:,} edges")
print(f"  GiG: {GiG_0.shape}, {GiG_0.nnz:,} edges")
print(f"  GpPW: {GpPW_0.shape}, {GpPW_0.nnz:,} edges")

# Get degrees
compound_degrees = np.array(CbG_0.sum(axis=1)).flatten()
gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()

# Compute pathway matrices
print("\nComputing pathway matrices...")
CbGiG_0 = CbG_0 @ GiG_0
GiGpPW_0 = GiG_0 @ GpPW_0
CbGiGpPW_0 = CbGiG_0 @ GpPW_0

print(f"  CbGiG: {CbGiG_0.nnz:,} non-zero")
print(f"  GiGpPW: {GiGpPW_0.nnz:,} non-zero")
print(f"  CbGiGpPW: {CbGiGpPW_0.nnz:,} non-zero")

# Train CbGiG model (reuse methodology from 2D)
print("\n" + "-"*80)
print("Training CbGiG model...")
print("-"*80)

np.random.seed(42)
sources_nz, targets_nz = CbGiG_0.nonzero()
n_nz = min(5000, len(sources_nz))
idx_nz = np.random.choice(len(sources_nz), n_nz, replace=False)
sampled_src = list(sources_nz[idx_nz])
sampled_tgt = list(targets_nz[idx_nz])

n_rand = 5000
sampled_src.extend(np.random.randint(0, CbGiG_0.shape[0], n_rand))
sampled_tgt.extend(np.random.randint(0, CbGiG_0.shape[1], n_rand))

X_cbgig = []
y_cbgig_train = []
CbGiG_0_lil = CbGiG_0.tolil()

for src, tgt in zip(sampled_src, sampled_tgt):
    features = extract_degree_features(compound_degrees[src], gene_degrees[tgt])
    X_cbgig.append(features)
    y_cbgig_train.append(CbGiG_0_lil[src, tgt])

X_cbgig = np.array(X_cbgig)
y_cbgig_train = np.array(y_cbgig_train, dtype=float)

# Load validation targets
val_counts = []
for perm_num in range(6, 21):
    if perm_num % 5 == 0:
        print(f"  Loading perm {perm_num}...")
    CbG_p = load_edge_matrix('CbG', perm_num=perm_num)
    GiG_p = load_edge_matrix('GiG', perm_num=perm_num)
    if CbG_p is None or GiG_p is None:
        continue
    CbGiG_p = (CbG_p @ GiG_p).tolil()
    counts = np.array([CbGiG_p[s, t] for s, t in zip(sampled_src, sampled_tgt)], dtype=float)
    val_counts.append(counts)

y_cbgig_val = np.mean(val_counts, axis=0)

X_train, X_test, y_train, y_test, y_val_train, y_val_test = train_test_split(
    X_cbgig, y_cbgig_train, y_cbgig_val, test_size=0.2, random_state=42
)

model_CbGiG = LinearRegression()
model_CbGiG.fit(X_train, y_train)
y_pred = model_CbGiG.predict(X_test)
r_cbgig = pearsonr(y_pred, y_val_test)[0]

print(f"  CbGiG model: r = {r_cbgig:.4f}")

# Train GiGpPW model
print("\n" + "-"*80)
print("Training GiGpPW count model...")
print("-"*80)

sources_nz_gp, targets_nz_gp = GiGpPW_0.nonzero()
n_nz_gp = min(5000, len(sources_nz_gp))
idx_nz_gp = np.random.choice(len(sources_nz_gp), n_nz_gp, replace=False)
sampled_src_gp = list(sources_nz_gp[idx_nz_gp])
sampled_tgt_gp = list(targets_nz_gp[idx_nz_gp])

sampled_src_gp.extend(np.random.randint(0, GiGpPW_0.shape[0], 5000))
sampled_tgt_gp.extend(np.random.randint(0, GiGpPW_0.shape[1], 5000))

X_gigppw = []
y_gigppw_train = []
GiGpPW_0_lil = GiGpPW_0.tolil()

for src, tgt in zip(sampled_src_gp, sampled_tgt_gp):
    features = extract_degree_features(gene_degrees[src], pathway_degrees[tgt])
    X_gigppw.append(features)
    y_gigppw_train.append(GiGpPW_0_lil[src, tgt])

X_gigppw = np.array(X_gigppw)
y_gigppw_train = np.array(y_gigppw_train, dtype=float)

# Load validation targets
val_counts_gp = []
for perm_num in range(6, 21):
    if perm_num % 5 == 0:
        print(f"  Loading perm {perm_num}...")
    GiG_p = load_edge_matrix('GiG', perm_num=perm_num)
    GpPW_p = load_edge_matrix('GpPW', perm_num=perm_num)
    if GiG_p is None or GpPW_p is None:
        continue
    GiGpPW_p = (GiG_p @ GpPW_p).tolil()
    counts = np.array([GiGpPW_p[s, t] for s, t in zip(sampled_src_gp, sampled_tgt_gp)], dtype=float)
    val_counts_gp.append(counts)

y_gigppw_val = np.mean(val_counts_gp, axis=0)

X_train_gp, X_test_gp, y_train_gp, y_test_gp, y_val_train_gp, y_val_test_gp = train_test_split(
    X_gigppw, y_gigppw_train, y_gigppw_val, test_size=0.2, random_state=42
)

model_GiGpPW = LinearRegression()
model_GiGpPW.fit(X_train_gp, y_train_gp)
y_pred_gp = model_GiGpPW.predict(X_test_gp)
r_gigppw = pearsonr(y_pred_gp, y_val_test_gp)[0]

print(f"  GiGpPW model: r = {r_gigppw:.4f}")

print(f"\nModels validated:")
print(f"  CbGiG: r = {r_cbgig:.4f} {'(SUCCESS)' if r_cbgig > 0.95 else '(WARNING: below threshold)'}")
print(f"  GiGpPW: r = {r_gigppw:.4f} {'(SUCCESS)' if r_gigppw > 0.95 else '(WARNING: below threshold)'}")


print("\n" + "="*80)
print("PHASE 2: Analyze Actual Pathway Decomposition by G2 Degree")
print("="*80)

print("\nSampling test pairs...")
np.random.seed(123)
sources_pw, targets_pw = CbGiGpPW_0.nonzero()
n_nz_pw = min(2500, len(sources_pw))
idx_pw = np.random.choice(len(sources_pw), n_nz_pw, replace=False)
test_src = list(sources_pw[idx_pw])
test_tgt = list(targets_pw[idx_pw])

test_src.extend(np.random.randint(0, CbGiGpPW_0.shape[0], 2500))
test_tgt.extend(np.random.randint(0, CbGiGpPW_0.shape[1], 2500))

print(f"  Sampled {len(test_src)} test pairs")

print("\nDecomposing actual pathways by G2 degree...")
decompositions = []
GpPW_0_csr = GpPW_0.tocsr()
CbGiGpPW_0_lil = CbGiGpPW_0.tolil()

for idx, (C, PW) in enumerate(zip(test_src[:100], test_tgt[:100])):  # Analyze first 100 for diagnostics
    if idx % 25 == 0:
        print(f"  Processed {idx}/100...")

    actual_total = CbGiGpPW_0_lil[C, PW]

    # Decompose by G2 degree
    contributions_by_G2_deg = defaultdict(float)
    G1_degrees_by_G2_deg = defaultdict(list)

    # Get genes connecting to PW
    genes_to_PW = GpPW_0_csr.getcol(PW).nonzero()[0]

    for G2 in genes_to_PW:
        deg_G2 = gene_degrees[G2]

        # This G2's contribution
        CbGiG_count = CbGiG_0_lil[C, G2]

        if CbGiG_count > 0:
            contributions_by_G2_deg[deg_G2] += CbGiG_count

            # For diagnostics: which G1 degrees contributed?
            G1_genes = CbG_0.getrow(C).nonzero()[1]
            for G1 in G1_genes:
                if GiG_0[G1, G2] > 0:
                    G1_degrees_by_G2_deg[deg_G2].append(gene_degrees[G1])

    decompositions.append({
        'C': C,
        'PW': PW,
        'deg_C': compound_degrees[C],
        'deg_PW': pathway_degrees[PW],
        'actual_total': actual_total,
        'contributions_by_G2_deg': dict(contributions_by_G2_deg),
        'n_G2_degrees': len(contributions_by_G2_deg),
        'G1_degrees_by_G2_deg': {k: list(v) for k, v in G1_degrees_by_G2_deg.items()}
    })

df_decomp = pd.DataFrame(decompositions)
df_decomp.to_csv(results_dir / 'experiment2f_decomposition_analysis.csv', index=False)

print(f"\nDecomposition statistics:")
print(f"  Avg G2 degrees contributing: {df_decomp['n_G2_degrees'].mean():.1f}")
print(f"  Max G2 degrees: {df_decomp['n_G2_degrees'].max()}")


print("\n" + "="*80)
print("PHASE 3: Degree-Stratified Composition")
print("="*80)

print("\nComputing empirical frequencies for G2→PW by degree...")
# Compute P(G2 degree d connects to PW degree p) from permutations
gpw_freq = defaultdict(lambda: {'count': 0, 'possible': 0})

for perm_num in range(1, 21):
    if perm_num % 5 == 0:
        print(f"  Processing perm {perm_num}...")

    GpPW_p = load_edge_matrix('GpPW', perm_num=perm_num)
    if GpPW_p is None:
        continue

    sources, targets = GpPW_p.nonzero()
    for s, t in zip(sources, targets):
        deg_g = gene_degrees[s]
        deg_p = pathway_degrees[t]
        gpw_freq[(deg_g, deg_p)]['count'] += 1

    # Count possible pairs
    for deg_g in np.unique(gene_degrees):
        n_g = np.sum(gene_degrees == deg_g)
        for deg_p in np.unique(pathway_degrees):
            n_p = np.sum(pathway_degrees == deg_p)
            gpw_freq[(deg_g, deg_p)]['possible'] += n_g * n_p

empirical_freq = {}
for key, val in gpw_freq.items():
    if val['possible'] > 0:
        empirical_freq[key] = val['count'] / val['possible']
    else:
        empirical_freq[key] = 0.0

print(f"  Computed frequencies for {len(empirical_freq)} degree pairs")

print("\nPerforming degree-stratified composition...")
y_true = []
y_pred = []
n_degrees_used = []

for idx, (C, PW) in enumerate(zip(test_src, test_tgt)):
    if idx % 1000 == 0:
        print(f"  Processed {idx}/{len(test_src)}...")

    actual_total = CbGiGpPW_0_lil[C, PW]

    deg_C = compound_degrees[C]
    deg_PW = pathway_degrees[PW]

    # Get genes connecting to PW
    genes_to_PW = GpPW_0_csr.getcol(PW).nonzero()[0]

    # Aggregate by G2 degree
    predicted_by_degree = {}

    for deg_G2 in np.unique(gene_degrees[genes_to_PW]):
        # Count how many G2 genes with this degree connect to PW
        n_G2 = np.sum(gene_degrees[genes_to_PW] == deg_G2)

        # Predict CbGiG count for this degree
        features_cbgig = extract_degree_features(deg_C, deg_G2)
        pred_CbGiG = model_CbGiG.predict([features_cbgig])[0]

        # Predict GiGpPW count for this degree
        features_gigppw = extract_degree_features(deg_G2, deg_PW)
        pred_GiGpPW = model_GiGpPW.predict([features_gigppw])[0]

        # Contribution from this degree
        # Option: use predicted counts directly
        contrib = n_G2 * pred_CbGiG * pred_GiGpPW

        predicted_by_degree[deg_G2] = contrib

    predicted_total = sum(predicted_by_degree.values())

    y_true.append(actual_total)
    y_pred.append(predicted_total)
    n_degrees_used.append(len(predicted_by_degree))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
n_degrees_used = np.array(n_degrees_used)

print(f"\n  Average G2 degrees used: {n_degrees_used.mean():.1f}")


print("\n" + "="*80)
print("PHASE 4: Evaluate and Diagnose")
print("="*80)

r = pearsonr(y_true, y_pred)[0]
mae = np.mean(np.abs(y_true - y_pred))
ratio = y_pred.mean() / y_true.mean() if y_true.mean() > 0 else 0

print(f"\nPerformance:")
print(f"  Correlation: r = {r:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  Mean true: {y_true.mean():.2f}")
print(f"  Mean predicted: {y_pred.mean():.2f}")
print(f"  Prediction ratio: {ratio:.3f}")

if r > 0.95:
    status = "SUCCESS"
elif r > 0.85:
    status = "PROMISING"
elif r > 0.70:
    status = "PARTIAL"
else:
    status = "FAILURE"

print(f"  Status: {status}")

print(f"\nComparison to baselines:")
print(f"  Exp 2E-v1 (actual CbGiG × freq): r = 0.855")
print(f"  Exp 2E-v2 (pred CbGiG × freq): r = 0.749")
print(f"  Exp 2F (degree-stratified counts): r = {r:.3f}")

# Save results
results = {
    'experiment': 'Experiment 2F',
    'description': 'Degree-stratified composition with count models',
    'r_cbgig': r_cbgig,
    'r_gigppw': r_gigppw,
    'r_composition': r,
    'mae': mae,
    'mean_true': y_true.mean(),
    'mean_pred': y_pred.mean(),
    'ratio': ratio,
    'avg_degrees_used': n_degrees_used.mean(),
    'status': status
}

pd.DataFrame([results]).to_csv(results_dir / 'experiment2f_results.csv', index=False)

print("\n" + "="*80)
print("EXPERIMENT 2F COMPLETE")
print("="*80)
print(f"\nFinal Status: {status}")
print(f"Correlation: r = {r:.4f}")
print(f"\nFiles saved:")
print(f"  {results_dir / 'experiment2f_results.csv'}")
print(f"  {results_dir / 'experiment2f_decomposition_analysis.csv'}")
