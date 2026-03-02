"""
Pair-level degree correction.

Regenerates baseline pair-level data and applies a linear correction model to reduce
residual bias against permutation-averaged pathway counts.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
XDG_CACHE_DIR = REPO_DIR / ".cache"
MPL_CACHE_DIR = XDG_CACHE_DIR / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pair-level degree correction.",
    )
    parser.add_argument("--edge1-type", type=str, default="CbG")
    parser.add_argument("--edge2-type", type=str, default="GpPW")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing edges/ and permutations/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "pair_level_phase2",
        help="Directory to save outputs.",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--perm-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit null permutations to average for the training target.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=20,
        help="Fallback number of available null permutations when --perm-ids is omitted.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def list_available_permutation_ids(data_dir: Path) -> list[int]:
    perm_dir = data_dir / "permutations"
    if not perm_dir.exists():
        return []
    perm_ids = []
    for child in perm_dir.glob("*.hetmat"):
        try:
            perm_ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(set(perm_ids))


def resolve_target_perms(
    available: list[int],
    requested: list[int] | None,
    n_permutations: int,
) -> list[int]:
    if not available:
        raise FileNotFoundError("No permutations found under data/permutations.")

    non_zero = [perm for perm in available if perm > 0]
    if requested:
        missing = [perm for perm in requested if perm not in available]
        if missing:
            raise FileNotFoundError(f"Requested permutations missing: {sorted(set(missing))}")
        return requested

    if not non_zero:
        # Fallback for very small local datasets that only have perm 0.
        return [available[0]]

    if n_permutations < 1:
        raise ValueError("--n-permutations must be >= 1")
    if n_permutations > len(non_zero):
        print(
            f"Requested {n_permutations} target permutations, but only {len(non_zero)} are available; "
            "using available set."
        )
    return non_zero[: min(n_permutations, len(non_zero))]


def load_edge_matrix(data_dir: Path, edge_type: str, perm_id: int | str = "original") -> sp.spmatrix:
    if perm_id == "original":
        edge_file = data_dir / "edges" / f"{edge_type}.sparse.npz"
    else:
        edge_file = (
            data_dir / "permutations" / f"{int(perm_id):03d}.hetmat" / "edges" / f"{edge_type}.sparse.npz"
        )
    if not edge_file.exists():
        raise FileNotFoundError(f"Missing edge file: {edge_file}")
    return sp.load_npz(edge_file).astype(np.int32)


def extract_pair_features_simple(
    edge1: sp.spmatrix,
    edge2: sp.spmatrix,
    source_idx: int,
    target_idx: int,
) -> np.ndarray:
    source_deg = edge1.getrow(source_idx).nnz
    target_deg = edge2.getcol(target_idx).nnz
    return np.array(
        [source_deg, target_deg, source_deg * target_deg, source_deg**2, target_deg**2],
        dtype=float,
    )


def sample_pairs_by_pathway_count(
    edge1: sp.spmatrix,
    edge2: sp.spmatrix,
    n_samples: int,
    random_state: int,
) -> list[tuple[int, int]]:
    np.random.seed(random_state)
    pathway_matrix = edge1 @ edge2
    sources, targets = pathway_matrix.nonzero()
    n_nonzero_target = min(int(n_samples * 0.5), len(sources))

    if len(sources) > n_nonzero_target:
        idx = np.random.choice(len(sources), n_nonzero_target, replace=False)
        sampled_sources = sources[idx]
        sampled_targets = targets[idx]
    else:
        sampled_sources = sources
        sampled_targets = targets

    n_random = n_samples - len(sampled_sources)
    if n_random > 0:
        zero_sources = np.random.randint(0, edge1.shape[0], n_random)
        zero_targets = np.random.randint(0, edge2.shape[1], n_random)
        sampled_sources = np.concatenate([sampled_sources, zero_sources])
        sampled_targets = np.concatenate([sampled_targets, zero_targets])

    return list(zip(sampled_sources, sampled_targets))


def compute_pathway_counts_for_pairs(
    edge1: sp.spmatrix,
    edge2: sp.spmatrix,
    pair_indices: list[tuple[int, int]],
) -> np.ndarray:
    pathway_matrix = edge1 @ edge2
    counts = np.zeros(len(pair_indices), dtype=float)
    for i, (source_idx, target_idx) in enumerate(pair_indices):
        counts[i] = pathway_matrix[source_idx, target_idx]
    return counts


def extract_correction_features(x_features: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    source_deg = x_features[:, 0]
    target_deg = x_features[:, 1]
    return np.column_stack(
        [
            source_deg,
            target_deg,
            source_deg * target_deg,
            source_deg**2,
            target_deg**2,
            np.sqrt(source_deg + 1),
            np.sqrt(target_deg + 1),
            y_pred,
            y_pred**2,
            np.log1p(np.maximum(y_pred, 0)),
            y_pred * source_deg,
            y_pred * target_deg,
            y_pred * source_deg * target_deg,
            np.sqrt(np.maximum(y_pred, 0) + 1) * source_deg,
            np.sqrt(np.maximum(y_pred, 0) + 1) * target_deg,
        ]
    )


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def main(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_samples = min(args.n_samples, 2_500)
        args.n_permutations = min(args.n_permutations, 4)

    available_perms = list_available_permutation_ids(data_dir)
    target_perms = resolve_target_perms(available_perms, args.perm_ids, args.n_permutations)
    base_perm = 0 if 0 in available_perms else target_perms[0]

    print("=" * 80)
    print("PAIR-LEVEL PHASE 2: DEGREE-AWARE CORRECTION")
    print("=" * 80)
    print(f"Edges: {args.edge1_type} + {args.edge2_type}")
    print(f"Target permutations: {target_perms}")
    print(f"Reference permutation: {base_perm:03d}")
    print(f"n_samples: {args.n_samples}")
    print()

    edge1_original = load_edge_matrix(data_dir, args.edge1_type, perm_id="original")
    edge2_original = load_edge_matrix(data_dir, args.edge2_type, perm_id="original")

    pair_indices = sample_pairs_by_pathway_count(
        edge1_original,
        edge2_original,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )
    print(f"Sampled {len(pair_indices)} pairs")

    x_features = np.array(
        [
            extract_pair_features_simple(edge1_original, edge2_original, source_idx, target_idx)
            for source_idx, target_idx in pair_indices
        ]
    )
    print(f"Feature matrix: {x_features.shape}")

    print("\nComputing target mean across selected permutations...")
    perm_counts = []
    for perm_id in target_perms:
        edge1_perm = load_edge_matrix(data_dir, args.edge1_type, perm_id)
        edge2_perm = load_edge_matrix(data_dir, args.edge2_type, perm_id)
        perm_counts.append(compute_pathway_counts_for_pairs(edge1_perm, edge2_perm, pair_indices))
    y_validation = np.mean(np.array(perm_counts), axis=0)
    print(f"Target mean: {y_validation.mean():.4f}")

    print("\nComputing reference permutation counts...")
    edge1_ref = load_edge_matrix(data_dir, args.edge1_type, perm_id=base_perm)
    edge2_ref = load_edge_matrix(data_dir, args.edge2_type, perm_id=base_perm)
    y_reference = compute_pathway_counts_for_pairs(edge1_ref, edge2_ref, pair_indices)

    base_model = LinearRegression()
    base_model.fit(x_features, y_validation)
    y_pred_base = base_model.predict(x_features)

    correction_features = extract_correction_features(x_features, y_pred_base)
    correction_target = y_reference - y_pred_base
    correction_model = LinearRegression()
    correction_model.fit(correction_features, correction_target)
    y_pred_corrected = y_pred_base + correction_model.predict(correction_features)

    r_base = safe_pearson(y_pred_base, y_validation)
    rmse_base = float(np.sqrt(np.mean((y_pred_base - y_validation) ** 2)))
    bias_base = float(np.mean(y_pred_base - y_validation))

    r_corrected = safe_pearson(y_pred_corrected, y_validation)
    rmse_corrected = float(np.sqrt(np.mean((y_pred_corrected - y_validation) ** 2)))
    bias_corrected = float(np.mean(y_pred_corrected - y_validation))

    print("\nResults:")
    print(f"  Baseline : r={r_base:.4f}, RMSE={rmse_base:.4f}, bias={bias_base:+.4f}")
    print(f"  Corrected: r={r_corrected:.4f}, RMSE={rmse_corrected:.4f}, bias={bias_corrected:+.4f}")
    print(f"  Delta r  : {r_corrected - r_base:+.4f}")
    print(f"  Delta RMSE: {rmse_corrected - rmse_base:+.4f}")

    if not args.skip_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(y_validation, y_pred_base, alpha=0.3, s=5)
        lim = float(max(np.max(y_validation), np.max(y_pred_base), 1.0))
        ax.plot([0, lim], [0, lim], "r--", label="Perfect prediction")
        ax.set_xlabel("True Pathway Count")
        ax.set_ylabel("Predicted Pathway Count")
        ax.set_title(f"Baseline (r = {r_base:.4f})")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.scatter(y_validation, y_pred_corrected, alpha=0.3, s=5)
        lim = float(max(np.max(y_validation), np.max(y_pred_corrected), 1.0))
        ax.plot([0, lim], [0, lim], "r--", label="Perfect prediction")
        ax.set_xlabel("True Pathway Count")
        ax.set_ylabel("Corrected Prediction")
        ax.set_title(f"Corrected (r = {r_corrected:.4f})")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        figure_path = results_dir / f"{args.edge1_type}{args.edge2_type}_phase2_correction.png"
        plt.savefig(figure_path, dpi=150)
        plt.close()
        print(f"Saved plot: {figure_path}")


if __name__ == "__main__":
    main(parse_args())
