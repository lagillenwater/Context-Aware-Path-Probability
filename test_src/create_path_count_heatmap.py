"""
Create path count heatmap visualization similar to Himmelstein et al. Figure 4.

This script generates heatmaps showing path counts stratified by source and target
node degree for the CbGpPWpG metapath, comparing unpermuted Hetionet to permuted
null networks.
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

warnings.filterwarnings("ignore")

METAPATH_EDGE_TYPES = ["CbG", "GpPW", "GpPW"]
TRANSPOSE_FLAGS = [False, False, True]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CbGpPWpG degree-stratified path count heatmaps.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing edges/ and permutations/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "path_count_visualization",
        help="Directory to write output figure.",
    )
    parser.add_argument(
        "--perm-ids",
        type=int,
        nargs="+",
        default=None,
        help="Permutation IDs to include. Defaults to first --num-perms available.",
    )
    parser.add_argument(
        "--num-perms",
        type=int,
        default=5,
        help="How many available permutations to use when --perm-ids is omitted.",
    )
    parser.add_argument("--min-source-deg", type=int, default=1)
    parser.add_argument("--max-source-deg", type=int, default=20)
    parser.add_argument("--min-target-deg", type=int, default=1)
    parser.add_argument("--max-target-deg", type=int, default=40)
    parser.add_argument(
        "--output-filename",
        type=str,
        default="CbGpPWpG_path_count_heatmap.png",
        help="Output figure filename.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use lighter defaults for quick validation.",
    )
    return parser.parse_args()


def list_available_permutation_ids(perm_dir: Path) -> list[int]:
    if not perm_dir.exists():
        return []
    perm_ids = []
    for child in perm_dir.glob("*.hetmat"):
        try:
            perm_ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(set(perm_ids))


def resolve_perm_ids(perm_dir: Path, requested: list[int] | None, num_perms: int) -> list[int]:
    available = list_available_permutation_ids(perm_dir)
    if not available:
        raise FileNotFoundError(f"No permutations found in {perm_dir}")

    if requested:
        missing = [perm_id for perm_id in requested if perm_id not in available]
        if missing:
            raise FileNotFoundError(
                f"Requested permutations missing from {perm_dir}: {missing}"
            )
        return requested

    if num_perms < 1:
        raise ValueError("--num-perms must be >= 1")

    if len(available) < num_perms:
        print(
            f"Requested {num_perms} permutations, but only {len(available)} are available; "
            f"using {available}."
        )
    return available[: min(num_perms, len(available))]


def load_edge_matrix(
    edge_type: str,
    data_dir: Path,
    perm_dir: Path,
    perm_id: int | None,
) -> sp.spmatrix:
    if perm_id is None:
        edge_file = data_dir / "edges" / f"{edge_type}.sparse.npz"
    else:
        edge_file = perm_dir / f"{perm_id:03d}.hetmat" / "edges" / f"{edge_type}.sparse.npz"

    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    return sp.load_npz(str(edge_file))


def compute_metapath_counts(edge_matrices: list[sp.spmatrix]) -> sp.spmatrix:
    result = edge_matrices[0].astype(float)
    for matrix in edge_matrices[1:]:
        result = result @ matrix.astype(float)
    return result


def compute_mean_per_pair_heatmap(
    path_counts: sp.spmatrix,
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
    min_source_deg: int,
    max_source_deg: int,
    min_target_deg: int,
    max_target_deg: int,
) -> np.ndarray:
    n_src_bins = max_source_deg - min_source_deg + 1
    n_tgt_bins = max_target_deg - min_target_deg + 1
    total_count = np.zeros((n_src_bins, n_tgt_bins))
    sum_paths = np.zeros((n_src_bins, n_tgt_bins))

    dense_counts = path_counts.toarray() if sp.issparse(path_counts) else path_counts

    for i, src_deg_val in enumerate(source_degrees):
        src_deg = int(src_deg_val)
        if src_deg < min_source_deg or src_deg > max_source_deg:
            continue
        src_idx = src_deg - min_source_deg

        for j, tgt_deg_val in enumerate(target_degrees):
            tgt_deg = int(tgt_deg_val)
            if tgt_deg < min_target_deg or tgt_deg > max_target_deg:
                continue
            tgt_idx = tgt_deg - min_target_deg
            total_count[src_idx, tgt_idx] += 1
            sum_paths[src_idx, tgt_idx] += dense_counts[i, j]

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total_count > 0, sum_paths / total_count, 0.0)


def main(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    perm_dir = data_dir / "permutations"
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.max_source_deg = min(args.max_source_deg, 5)
        args.max_target_deg = min(args.max_target_deg, 10)
        args.num_perms = min(args.num_perms, 3)

    perm_ids = resolve_perm_ids(perm_dir, args.perm_ids, args.num_perms)

    print("=" * 70)
    print("PATH COUNT HEATMAP VISUALIZATION")
    print("=" * 70)
    print("Metapath: CbGpPWpG (Compound -> Gene -> Pathway -> Gene)")
    print(f"Permutations: {perm_ids}\n")

    print("Loading unpermuted Hetionet...")
    hetionet_matrices: list[sp.spmatrix] = []
    for edge_type, needs_transpose in zip(METAPATH_EDGE_TYPES, TRANSPOSE_FLAGS):
        matrix = load_edge_matrix(edge_type, data_dir=data_dir, perm_dir=perm_dir, perm_id=None)
        if needs_transpose:
            matrix = matrix.T
            edge_label = f"{edge_type} (transposed)"
        else:
            edge_label = edge_type
        hetionet_matrices.append(matrix)
        print(f"  {edge_label}: {matrix.shape} ({matrix.nnz:,} edges)")

    hetionet_paths = compute_metapath_counts(hetionet_matrices)
    print(f"\nUnpermuted path counts: {hetionet_paths.sum():,.0f}")
    print(f"Non-zero pairs: {hetionet_paths.nnz:,}")

    source_degrees = np.array(hetionet_matrices[0].sum(axis=1)).flatten()
    target_degrees = np.array(hetionet_matrices[1].sum(axis=1)).flatten()

    print(
        f"\nSource degree range: {source_degrees.min()}-{source_degrees.max()}, "
        f"mean={source_degrees.mean():.1f}"
    )
    print(
        f"Target degree range: {target_degrees.min()}-{target_degrees.max()}, "
        f"mean={target_degrees.mean():.1f}"
    )

    print("\nLoading selected permutations...")
    perm_paths_list: list[sp.spmatrix] = []
    for perm_id in perm_ids:
        perm_matrices: list[sp.spmatrix] = []
        for edge_type, needs_transpose in zip(METAPATH_EDGE_TYPES, TRANSPOSE_FLAGS):
            matrix = load_edge_matrix(edge_type, data_dir=data_dir, perm_dir=perm_dir, perm_id=perm_id)
            if needs_transpose:
                matrix = matrix.T
            perm_matrices.append(matrix)

        perm_paths = compute_metapath_counts(perm_matrices)
        perm_paths_list.append(perm_paths)
        print(f"  Perm {perm_id:03d}: total={perm_paths.sum():,.0f}, nnz={perm_paths.nnz:,}")

    print("\nComputing heatmaps...")
    perm_mean_per_pair_list = []
    for perm_paths in perm_paths_list:
        mean_per_pair = compute_mean_per_pair_heatmap(
            perm_paths,
            source_degrees,
            target_degrees,
            min_source_deg=args.min_source_deg,
            max_source_deg=args.max_source_deg,
            min_target_deg=args.min_target_deg,
            max_target_deg=args.max_target_deg,
        )
        perm_mean_per_pair_list.append(mean_per_pair)

    perm_means_3d = np.stack(perm_mean_per_pair_list, axis=0)
    mean_of_means = np.mean(perm_means_3d, axis=0)
    variance_of_means = np.var(perm_means_3d, axis=0)
    std_of_means = np.std(perm_means_3d, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean_of_means > 0, std_of_means / mean_of_means, 0.0)

    n_src_bins = args.max_source_deg - args.min_source_deg + 1
    n_tgt_bins = args.max_target_deg - args.min_target_deg + 1
    pair_counts = np.zeros((n_src_bins, n_tgt_bins))

    for src_deg_val in source_degrees:
        src_deg = int(src_deg_val)
        if src_deg < args.min_source_deg or src_deg > args.max_source_deg:
            continue
        src_idx = src_deg - args.min_source_deg
        for tgt_deg_val in target_degrees:
            tgt_deg = int(tgt_deg_val)
            if tgt_deg < args.min_target_deg or tgt_deg > args.max_target_deg:
                continue
            tgt_idx = tgt_deg - args.min_target_deg
            pair_counts[src_idx, tgt_idx] += 1

    total_counts = mean_of_means * pair_counts

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    tick_positions_x = range(0, args.max_target_deg - args.min_target_deg + 1, 5)
    tick_labels_x = range(args.min_target_deg, args.max_target_deg + 1, 5)
    tick_positions_y = range(0, args.max_source_deg - args.min_source_deg + 1, 5)
    tick_labels_y = range(args.min_source_deg, args.max_source_deg + 1, 5)

    ax_topleft = axes[0, 0]
    im_topleft = ax_topleft.imshow(
        total_counts,
        aspect="auto",
        origin="lower",
        cmap="cividis",
        interpolation="nearest",
    )
    ax_topleft.set_title(f"Permutations {perm_ids}\nTotal Path Counts", fontsize=14, fontweight="bold")
    ax_topleft.set_xlabel("Target Gene Degree", fontsize=12)
    ax_topleft.set_ylabel("Source Compound Degree", fontsize=12)
    ax_topleft.set_xticks(tick_positions_x)
    ax_topleft.set_xticklabels(tick_labels_x)
    ax_topleft.set_yticks(tick_positions_y)
    ax_topleft.set_yticklabels(tick_labels_y)
    plt.colorbar(im_topleft, ax=ax_topleft, label="Total Counts")

    ax_topright = axes[0, 1]
    im_topright = ax_topright.imshow(
        mean_of_means,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    ax_topright.set_title(
        f"Permutations {perm_ids}\nMean Path Count per Pair",
        fontsize=14,
        fontweight="bold",
    )
    ax_topright.set_xlabel("Target Gene Degree", fontsize=12)
    ax_topright.set_ylabel("Source Compound Degree", fontsize=12)
    ax_topright.set_xticks(tick_positions_x)
    ax_topright.set_xticklabels(tick_labels_x)
    ax_topright.set_yticks(tick_positions_y)
    ax_topright.set_yticklabels(tick_labels_y)
    plt.colorbar(im_topright, ax=ax_topright, label="Mean Count per Pair")

    ax_bottomleft = axes[1, 0]
    im_bottomleft = ax_bottomleft.imshow(
        variance_of_means,
        aspect="auto",
        origin="lower",
        cmap="plasma",
        interpolation="nearest",
    )
    ax_bottomleft.set_title(
        f"Permutations {perm_ids}\nVariance of Mean Count per Pair",
        fontsize=14,
        fontweight="bold",
    )
    ax_bottomleft.set_xlabel("Target Gene Degree", fontsize=12)
    ax_bottomleft.set_ylabel("Source Compound Degree", fontsize=12)
    ax_bottomleft.set_xticks(tick_positions_x)
    ax_bottomleft.set_xticklabels(tick_labels_x)
    ax_bottomleft.set_yticks(tick_positions_y)
    ax_bottomleft.set_yticklabels(tick_labels_y)
    plt.colorbar(im_bottomleft, ax=ax_bottomleft, label="Variance")

    ax_bottomright = axes[1, 1]
    im_bottomright = ax_bottomright.imshow(
        cv,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    ax_bottomright.set_title(
        f"Permutations {perm_ids}\nCoefficient of Variation",
        fontsize=14,
        fontweight="bold",
    )
    ax_bottomright.set_xlabel("Target Gene Degree", fontsize=12)
    ax_bottomright.set_ylabel("Source Compound Degree", fontsize=12)
    ax_bottomright.set_xticks(tick_positions_x)
    ax_bottomright.set_xticklabels(tick_labels_x)
    ax_bottomright.set_yticks(tick_positions_y)
    ax_bottomright.set_yticklabels(tick_labels_y)
    plt.colorbar(im_bottomright, ax=ax_bottomright, label="CV (std/mean)")

    plt.tight_layout()
    output_file = results_dir / args.output_filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {output_file}")


if __name__ == "__main__":
    main(parse_args())
