"""
Check if intermediate counts are preserved across XSwap permutations.

For 3-hop path CbGiGpPW, check if the number of intermediate genes
connecting compound C to pathway PW is the same across permutations.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'

def load_edge_matrix(edge_abbrev, perm_num):
    edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'
    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    return None

print("Loading permutations 0, 1, 2...")
CbG_0 = load_edge_matrix('CbG', 0)
GiG_0 = load_edge_matrix('GiG', 0)
GpPW_0 = load_edge_matrix('GpPW', 0)

CbG_1 = load_edge_matrix('CbG', 1)
GiG_1 = load_edge_matrix('GiG', 1)
GpPW_1 = load_edge_matrix('GpPW', 1)

CbG_2 = load_edge_matrix('CbG', 2)
GiG_2 = load_edge_matrix('GiG', 2)
GpPW_2 = load_edge_matrix('GpPW', 2)

# Compute 2-hop paths
CbGiG_0 = CbG_0 @ GiG_0
CbGiG_1 = CbG_1 @ GiG_1
CbGiG_2 = CbG_2 @ GiG_2

# Compute 3-hop paths
CbGiGpPW_0 = CbGiG_0 @ GpPW_0
CbGiGpPW_1 = CbGiG_1 @ GpPW_1
CbGiGpPW_2 = CbGiG_2 @ GpPW_2

print("\nChecking intermediate counts for sample pairs...")

# Convert to efficient formats
CbGiG_0_lil = CbGiG_0.tolil()
CbGiG_1_lil = CbGiG_1.tolil()
CbGiG_2_lil = CbGiG_2.tolil()
GpPW_0_csr = GpPW_0.tocsr()
GpPW_1_csr = GpPW_1.tocsr()
GpPW_2_csr = GpPW_2.tocsr()

# Sample pairs with pathways in perm 0
CbGiGpPW_0_coo = CbGiGpPW_0.tocoo()
n_sample = min(100, len(CbGiGpPW_0_coo.data))
sample_idx = np.random.choice(len(CbGiGpPW_0_coo.data), n_sample, replace=False)
pairs = [(CbGiGpPW_0_coo.row[i], CbGiGpPW_0_coo.col[i]) for i in sample_idx]

differences = []
for C, PW in pairs:
    # Count intermediates in each permutation
    genes_to_PW_0 = GpPW_0_csr[:, PW].nonzero()[0]
    intermediates_0 = [G for G in genes_to_PW_0 if CbGiG_0_lil[C, G] > 0]

    genes_to_PW_1 = GpPW_1_csr[:, PW].nonzero()[0]
    intermediates_1 = [G for G in genes_to_PW_1 if CbGiG_1_lil[C, G] > 0]

    genes_to_PW_2 = GpPW_2_csr[:, PW].nonzero()[0]
    intermediates_2 = [G for G in genes_to_PW_2 if CbGiG_2_lil[C, G] > 0]

    n0 = len(intermediates_0)
    n1 = len(intermediates_1)
    n2 = len(intermediates_2)

    differences.append((n0, n1, n2))

differences = np.array(differences)

print(f"\nIntermediate counts across {n_sample} pairs:")
print(f"  Perm 0 mean: {differences[:, 0].mean():.2f} (std: {differences[:, 0].std():.2f})")
print(f"  Perm 1 mean: {differences[:, 1].mean():.2f} (std: {differences[:, 1].std():.2f})")
print(f"  Perm 2 mean: {differences[:, 2].mean():.2f} (std: {differences[:, 2].std():.2f})")

print(f"\nPairwise differences:")
print(f"  |n0 - n1| mean: {np.abs(differences[:, 0] - differences[:, 1]).mean():.2f}")
print(f"  |n0 - n2| mean: {np.abs(differences[:, 0] - differences[:, 2]).mean():.2f}")
print(f"  |n1 - n2| mean: {np.abs(differences[:, 1] - differences[:, 2]).mean():.2f}")

# Check if ANY pairs have identical counts
same_01 = np.sum(differences[:, 0] == differences[:, 1])
same_02 = np.sum(differences[:, 0] == differences[:, 2])
same_12 = np.sum(differences[:, 1] == differences[:, 2])

print(f"\nPairs with identical counts:")
print(f"  Perm 0 == Perm 1: {same_01}/{n_sample} ({100*same_01/n_sample:.1f}%)")
print(f"  Perm 0 == Perm 2: {same_02}/{n_sample} ({100*same_02/n_sample:.1f}%)")
print(f"  Perm 1 == Perm 2: {same_12}/{n_sample} ({100*same_12/n_sample:.1f}%)")

# Check correlation
from scipy.stats import pearsonr
r_01 = pearsonr(differences[:, 0], differences[:, 1])[0]
r_02 = pearsonr(differences[:, 0], differences[:, 2])[0]
r_12 = pearsonr(differences[:, 1], differences[:, 2])[0]

print(f"\nCorrelation of intermediate counts:")
print(f"  r(perm0, perm1): {r_01:.4f}")
print(f"  r(perm0, perm2): {r_02:.4f}")
print(f"  r(perm1, perm2): {r_12:.4f}")

print("\nConclusion:")
if r_01 > 0.95 and r_02 > 0.95:
    print("  Intermediate counts are HIGHLY PRESERVED (r > 0.95)")
    print("  XSwap maintains approximate intermediate structure")
elif r_01 > 0.8 and r_02 > 0.8:
    print("  Intermediate counts are MODERATELY PRESERVED (r > 0.8)")
    print("  Counts vary but are correlated across permutations")
else:
    print("  Intermediate counts are NOT PRESERVED (r < 0.8)")
    print("  Different permutations have different intermediate structures")
