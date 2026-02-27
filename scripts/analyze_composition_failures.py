"""Analyze compositional prediction failures.

- stratified residual sampling by source/target degree bins
- degree-stratified correlation diagnostics
- linear correction analysis
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


REPO_DIR = Path(__file__).resolve().parents[1]


@dataclass
class MetapathSpec:
    name: str
    edge1: str
    edge2: str
    description: str
    transpose_edge1: bool = False
    transpose_edge2: bool = False


DEFAULT_METAPATHS: dict[str, MetapathSpec] = {
    "CbGpPW": MetapathSpec(
        name="CbGpPW",
        edge1="CbG",
        edge2="GpPW",
        description="Compound-binds-Gene-participates-Pathway",
    ),
    "CtDaG": MetapathSpec(
        name="CtDaG",
        edge1="CtD",
        edge2="DaG",
        description="Compound-treats-Disease-associates-Gene",
    ),
    "CbGaD": MetapathSpec(
        name="CbGaD",
        edge1="CbG",
        edge2="DaG",
        transpose_edge2=True,
        description="Compound-binds-Gene-associates-Disease",
    ),
    "CrCbG": MetapathSpec(
        name="CrCbG",
        edge1="CrC",
        edge2="CbG",
        description="Compound-resembles-Compound-binds-Gene",
    ),
    "CbGiG": MetapathSpec(
        name="CbGiG",
        edge1="CbG",
        edge2="GiG",
        description="Compound-binds-Gene-interacts-Gene",
    ),
    "CpDaG": MetapathSpec(
        name="CpDaG",
        edge1="CpD",
        edge2="DaG",
        description="Compound-palliates-Disease-associates-Gene",
    ),
    "CbGpBP": MetapathSpec(
        name="CbGpBP",
        edge1="CbG",
        edge2="GpBP",
        description="Compound-binds-Gene-participates-BiologicalProcess",
    ),
    "CbGpCC": MetapathSpec(
        name="CbGpCC",
        edge1="CbG",
        edge2="GpCC",
        description="Compound-binds-Gene-participates-CellularComponent",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze compositional failures with degree-stratified residual "
            "sampling and correction diagnostics."
        )
    )
    parser.add_argument("--train-perms-start", type=int, default=1)
    parser.add_argument("--train-perms-end", type=int, default=20)
    parser.add_argument("--valid-perms-start", type=int, default=21)
    parser.add_argument("--valid-perms-end", type=int, default=30)
    parser.add_argument(
        "--metapath",
        action="append",
        default=[],
        help=(
            "Metapath name to analyze (repeatable). "
            f"Available: {', '.join(DEFAULT_METAPATHS.keys())}"
        ),
    )
    parser.add_argument(
        "--use-validation-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled and results/compositional_validation/validation_summary.json exists, "
            "use its metapath list by default."
        ),
    )
    parser.add_argument("--n-degree-bins", type=int, default=10)
    parser.add_argument("--samples-per-bin", type=int, default=200)
    parser.add_argument(
        "--max-locations",
        type=int,
        default=None,
        help="Optional cap on candidate (source,target) locations per metapath.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--permutations-dir",
        type=Path,
        default=REPO_DIR / "data" / "permutations",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "compositional_validation",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip writing diagnostic plots.",
    )
    parser.add_argument(
        "--clear-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete existing failure_analysis.csv before writing. Default: true.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue remaining metapaths if one fails. Default: true.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.train_perms_start > args.train_perms_end:
        raise ValueError("train-perms-start must be <= train-perms-end")
    if args.valid_perms_start > args.valid_perms_end:
        raise ValueError("valid-perms-start must be <= valid-perms-end")
    if args.n_degree_bins <= 1:
        raise ValueError("n-degree-bins must be > 1")
    if args.samples_per_bin <= 0:
        raise ValueError("samples-per-bin must be > 0")
    if args.max_locations is not None and args.max_locations <= 0:
        raise ValueError("max-locations must be > 0")


def resolve_metapaths(args: argparse.Namespace) -> list[MetapathSpec]:
    selected_names: list[str] = []

    if args.metapath:
        selected_names = args.metapath
    elif args.use_validation_summary:
        summary_file = args.results_dir / "validation_summary.json"
        if summary_file.exists():
            try:
                payload = json.loads(summary_file.read_text())
                selected_names = [row["metapath"] for row in payload.get("metapath_results", [])]
            except Exception:
                selected_names = []

    if not selected_names:
        selected_names = list(DEFAULT_METAPATHS.keys())

    resolved: list[MetapathSpec] = []
    for name in selected_names:
        if name not in DEFAULT_METAPATHS:
            print(f"WARNING: skipping unknown metapath '{name}'")
            continue
        resolved.append(DEFAULT_METAPATHS[name])
    if not resolved:
        raise RuntimeError("No valid metapaths selected.")
    return resolved


def load_edge_matrix(
    *,
    permutations_dir: Path,
    perm_id: int,
    edge_type: str,
    transpose: bool = False,
) -> sp.csr_matrix:
    edge_file = permutations_dir / f"{perm_id:03d}.hetmat" / "edges" / f"{edge_type}.sparse.npz"
    if not edge_file.exists():
        raise FileNotFoundError(f"Missing edge file: {edge_file}")
    matrix = sp.load_npz(str(edge_file)).tocsr()
    return matrix.transpose().tocsr() if transpose else matrix


def compute_empirical_edge_probs(
    *,
    permutations_dir: Path,
    edge_type: str,
    perm_ids: list[int],
    transpose: bool = False,
) -> sp.csr_matrix:
    first = load_edge_matrix(
        permutations_dir=permutations_dir,
        perm_id=perm_ids[0],
        edge_type=edge_type,
        transpose=transpose,
    )
    edge_sum = sp.csr_matrix(first.shape, dtype=np.float64)
    for perm_id in perm_ids:
        edge_sum = edge_sum + load_edge_matrix(
            permutations_dir=permutations_dir,
            perm_id=perm_id,
            edge_type=edge_type,
            transpose=transpose,
        ).astype(np.float64)
    return (edge_sum / len(perm_ids)).tocsr()


def create_degree_bins(degrees: np.ndarray, n_bins: int) -> np.ndarray:
    nonzero = degrees[degrees > 0]
    if len(nonzero) == 0:
        return np.array([0.0, 1.0])
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(nonzero, quantiles)
    bins[0] = 0.0
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.array([0.0, float(nonzero.max()) + 1.0])
    return bins


def assign_to_bins(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(values, bin_edges, right=False) - 1
    return np.clip(idx, 0, len(bin_edges) - 2).astype(int)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        r, _ = pearsonr(x, y)
        return float(r)
    except Exception:
        return np.nan


def clean_num(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def maybe_write_plots(
    *,
    residuals_df: pd.DataFrame,
    metapaths: list[MetapathSpec],
    correction_df: pd.DataFrame,
    degree_strat_df: pd.DataFrame,
    results_dir: Path,
    random_seed: int,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    names = [m.name for m in metapaths]

    # Residual distributions
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, name in enumerate(names):
        ax = axes[idx]
        mp_data = residuals_df[residuals_df["metapath"] == name]
        if mp_data.empty:
            ax.set_visible(False)
            continue
        ax.hist(mp_data["residual"], bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(name, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / "residual_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Actual vs predicted
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, name in enumerate(names):
        ax = axes[idx]
        mp_data = residuals_df[residuals_df["metapath"] == name]
        if mp_data.empty:
            ax.set_visible(False)
            continue
        plot_data = mp_data.sample(n=min(5000, len(mp_data)), random_state=random_seed)
        ax.scatter(plot_data["predicted"], plot_data["actual"], alpha=0.25, s=5)
        max_val = max(float(plot_data["predicted"].max()), float(plot_data["actual"].max()))
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5)
        corr_row = degree_strat_df[degree_strat_df["metapath"] == name]
        corr = corr_row["overall_r"].iloc[0] if not corr_row.empty else np.nan
        ax.set_title(f"{name} (r={corr:.3f})", fontweight="bold")
        ax.grid(alpha=0.3)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Heatmaps by (source_bin, target_bin)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, name in enumerate(names):
        ax = axes[idx]
        mp_data = residuals_df[residuals_df["metapath"] == name]
        if mp_data.empty:
            ax.set_visible(False)
            continue
        n_src = int(mp_data["source_bin"].max()) + 1
        n_tgt = int(mp_data["target_bin"].max()) + 1
        corr_mat = np.full((n_src, n_tgt), np.nan)
        for sbin in range(n_src):
            for tbin in range(n_tgt):
                bin_data = mp_data[(mp_data["source_bin"] == sbin) & (mp_data["target_bin"] == tbin)]
                if len(bin_data) > 10:
                    corr_mat[sbin, tbin] = safe_pearson(
                        bin_data["predicted"].to_numpy(),
                        bin_data["actual"].to_numpy(),
                    )
        im = ax.imshow(corr_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Target Degree Bin")
        ax.set_ylabel("Source Degree Bin")
    for ax in axes[len(names):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / "failure_heatmaps_by_degree.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Linear correction impact
    if not correction_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(correction_df))
        width = 0.35
        ax.bar(x - width / 2, correction_df["original_r"], width, label="Original", alpha=0.75, color="red")
        ax.bar(
            x + width / 2,
            correction_df["corrected_r"],
            width,
            label="After Linear Correction",
            alpha=0.75,
            color="green",
        )
        ax.axhline(0.85, color="blue", linestyle="--", linewidth=2, label="Required (r=0.85)")
        ax.set_xticks(x)
        ax.set_xticklabels(correction_df["metapath"], rotation=45, ha="right")
        ax.set_ylabel("Pearson Correlation")
        ax.set_title("Linear Correction Impact")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "linear_correction_impact.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "plots").mkdir(parents=True, exist_ok=True)

    metapaths = resolve_metapaths(args)
    train_perm_ids = list(range(args.train_perms_start, args.train_perms_end + 1))
    valid_perm_ids = list(range(args.valid_perms_start, args.valid_perms_end + 1))
    rng = np.random.default_rng(args.random_seed)

    print(f"Results directory: {args.results_dir}")
    print(f"Metapaths: {[m.name for m in metapaths]}")
    print(f"Training perms: {train_perm_ids[0]}-{train_perm_ids[-1]}")
    print(f"Validation perms: {valid_perm_ids[0]}-{valid_perm_ids[-1]}")
    print(f"Degree bins: {args.n_degree_bins}; samples/bin: {args.samples_per_bin}")

    csv_file = args.results_dir / "failure_analysis.csv"
    if args.clear_output and csv_file.exists():
        csv_file.unlink()
        print(f"Deleted existing {csv_file}")

    write_header = True
    edge_prob_cache: dict[tuple[str, bool], sp.csr_matrix] = {}

    for idx, spec in enumerate(metapaths, start=1):
        try:
            print("\n" + "=" * 70)
            print(f"Analyzing {spec.name} ({idx}/{len(metapaths)}): {spec.edge1} -> {spec.edge2}")
            print("=" * 70)

            key1 = (spec.edge1, spec.transpose_edge1)
            key2 = (spec.edge2, spec.transpose_edge2)
            if key1 not in edge_prob_cache:
                edge_prob_cache[key1] = compute_empirical_edge_probs(
                    permutations_dir=args.permutations_dir,
                    edge_type=spec.edge1,
                    perm_ids=train_perm_ids,
                    transpose=spec.transpose_edge1,
                )
            if key2 not in edge_prob_cache:
                edge_prob_cache[key2] = compute_empirical_edge_probs(
                    permutations_dir=args.permutations_dir,
                    edge_type=spec.edge2,
                    perm_ids=train_perm_ids,
                    transpose=spec.transpose_edge2,
                )

            edge1_probs = edge_prob_cache[key1]
            edge2_probs = edge_prob_cache[key2]
            if edge1_probs.shape[1] != edge2_probs.shape[0]:
                raise ValueError(
                    f"Shape mismatch for {spec.name}: {edge1_probs.shape} then {edge2_probs.shape}"
                )

            predicted = (edge1_probs @ edge2_probs).tocsr()
            pred_coo = predicted.tocoo()
            pred_pairs = list(zip(pred_coo.row.tolist(), pred_coo.col.tolist()))
            pred_vals = pred_coo.data.astype(float)
            print(f"Predicted non-zero entries: {len(pred_pairs):,}")

            # degree vectors from first train perm
            edge1_train = load_edge_matrix(
                permutations_dir=args.permutations_dir,
                perm_id=train_perm_ids[0],
                edge_type=spec.edge1,
                transpose=spec.transpose_edge1,
            )
            edge2_train = load_edge_matrix(
                permutations_dir=args.permutations_dir,
                perm_id=train_perm_ids[0],
                edge_type=spec.edge2,
                transpose=spec.transpose_edge2,
            )
            source_degrees = np.asarray(edge1_train.sum(axis=1)).ravel()
            target_degrees = np.asarray(edge2_train.sum(axis=0)).ravel()

            source_bins = create_degree_bins(source_degrees, args.n_degree_bins)
            target_bins = create_degree_bins(target_degrees, args.n_degree_bins)

            # union with one validation permutation
            actual_sample = (
                load_edge_matrix(
                    permutations_dir=args.permutations_dir,
                    perm_id=valid_perm_ids[0],
                    edge_type=spec.edge1,
                    transpose=spec.transpose_edge1,
                )
                @ load_edge_matrix(
                    permutations_dir=args.permutations_dir,
                    perm_id=valid_perm_ids[0],
                    edge_type=spec.edge2,
                    transpose=spec.transpose_edge2,
                )
            ).tocoo()
            actual_sample_pairs = list(zip(actual_sample.row.tolist(), actual_sample.col.tolist()))

            all_locations = list(set(pred_pairs) | set(actual_sample_pairs))
            if args.max_locations is not None and len(all_locations) > args.max_locations:
                sample_idx = rng.choice(len(all_locations), size=args.max_locations, replace=False)
                all_locations = [all_locations[int(i)] for i in sample_idx]
            print(f"Candidate locations: {len(all_locations):,}")

            pred_dict = {pair: float(v) for pair, v in zip(pred_pairs, pred_vals)}

            # assign to bins
            location_bins: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for i_loc, j_loc in all_locations:
                src_bin = int(assign_to_bins(np.array([source_degrees[i_loc]]), source_bins)[0])
                tgt_bin = int(assign_to_bins(np.array([target_degrees[j_loc]]), target_bins)[0])
                location_bins.setdefault((src_bin, tgt_bin), []).append((int(i_loc), int(j_loc)))

            sampled_pairs: list[tuple[int, int, int, int]] = []
            for (src_bin, tgt_bin), pairs in location_bins.items():
                n_take = min(args.samples_per_bin, len(pairs))
                if n_take < len(pairs):
                    choose_idx = rng.choice(len(pairs), size=n_take, replace=False)
                    chosen = [pairs[int(i)] for i in choose_idx]
                else:
                    chosen = pairs
                sampled_pairs.extend((src_bin, tgt_bin, i_loc, j_loc) for i_loc, j_loc in chosen)

            print(f"Sampled pairs: {len(sampled_pairs):,}")

            rows: list[dict[str, float | int | str]] = []
            for perm_id in valid_perm_ids:
                actual = (
                    load_edge_matrix(
                        permutations_dir=args.permutations_dir,
                        perm_id=perm_id,
                        edge_type=spec.edge1,
                        transpose=spec.transpose_edge1,
                    )
                    @ load_edge_matrix(
                        permutations_dir=args.permutations_dir,
                        perm_id=perm_id,
                        edge_type=spec.edge2,
                        transpose=spec.transpose_edge2,
                    )
                ).tocoo()
                actual_dict = {
                    (int(i), int(j)): float(v)
                    for i, j, v in zip(actual.row.tolist(), actual.col.tolist(), actual.data.tolist())
                }
                for src_bin, tgt_bin, i_loc, j_loc in sampled_pairs:
                    pred_val = float(pred_dict.get((i_loc, j_loc), 0.0))
                    actual_val = float(actual_dict.get((i_loc, j_loc), 0.0))
                    residual = actual_val - pred_val
                    rows.append(
                        {
                            "metapath": spec.name,
                            "perm_id": int(perm_id),
                            "source_id": int(i_loc),
                            "target_id": int(j_loc),
                            "source_degree": int(source_degrees[i_loc]),
                            "target_degree": int(target_degrees[j_loc]),
                            "source_bin": int(src_bin),
                            "target_bin": int(tgt_bin),
                            "predicted": pred_val,
                            "actual": actual_val,
                            "residual": residual,
                            "abs_residual": float(abs(residual)),
                            "pct_error": float((residual / pred_val * 100) if pred_val > 0 else np.nan),
                        }
                    )

            chunk_df = pd.DataFrame(rows)
            chunk_df.to_csv(csv_file, mode="a", header=write_header, index=False)
            write_header = False
            print(f"Wrote {len(chunk_df):,} rows to {csv_file}")

        except Exception as exc:
            msg = f"Failed metapath {spec.name}: {exc}"
            if args.continue_on_error:
                print(f"WARNING: {msg}")
                continue
            raise RuntimeError(msg) from exc

    if not csv_file.exists():
        raise RuntimeError("No failure analysis CSV produced.")

    residuals_df = pd.read_csv(
        csv_file,
        dtype={
            "metapath": "str",
            "perm_id": "int64",
            "source_id": "int64",
            "target_id": "int64",
            "source_degree": "int64",
            "target_degree": "int64",
            "source_bin": "int64",
            "target_bin": "int64",
            "predicted": "float64",
            "actual": "float64",
            "residual": "float64",
            "abs_residual": "float64",
            "pct_error": "float64",
        },
    )
    print(f"\nLoaded residuals: {len(residuals_df):,} rows")

    analyzed_names = [m.name for m in metapaths if m.name in set(residuals_df["metapath"].unique())]

    # Degree-stratified correlations
    degree_rows: list[dict[str, float | str | None]] = []
    for name in analyzed_names:
        mp_data = residuals_df[residuals_df["metapath"] == name]
        overall_r = safe_pearson(mp_data["predicted"].to_numpy(), mp_data["actual"].to_numpy())

        source_bin_corrs = []
        for b in sorted(mp_data["source_bin"].unique()):
            bin_data = mp_data[mp_data["source_bin"] == b]
            if len(bin_data) > 10:
                source_bin_corrs.append(
                    safe_pearson(bin_data["predicted"].to_numpy(), bin_data["actual"].to_numpy())
                )
        target_bin_corrs = []
        for b in sorted(mp_data["target_bin"].unique()):
            bin_data = mp_data[mp_data["target_bin"] == b]
            if len(bin_data) > 10:
                target_bin_corrs.append(
                    safe_pearson(bin_data["predicted"].to_numpy(), bin_data["actual"].to_numpy())
                )

        joint_corrs = []
        for sb in sorted(mp_data["source_bin"].unique()):
            for tb in sorted(mp_data["target_bin"].unique()):
                bin_data = mp_data[(mp_data["source_bin"] == sb) & (mp_data["target_bin"] == tb)]
                if len(bin_data) > 10:
                    joint_corrs.append(
                        safe_pearson(bin_data["predicted"].to_numpy(), bin_data["actual"].to_numpy())
                    )

        mean_source = float(np.nanmean(source_bin_corrs)) if source_bin_corrs else np.nan
        mean_target = float(np.nanmean(target_bin_corrs)) if target_bin_corrs else np.nan
        mean_joint = float(np.nanmean(joint_corrs)) if joint_corrs else np.nan

        degree_rows.append(
            {
                "metapath": name,
                "overall_r": clean_num(overall_r),
                "mean_source_bin_r": clean_num(mean_source),
                "mean_target_bin_r": clean_num(mean_target),
                "mean_joint_bin_r": clean_num(mean_joint),
                "source_improvement": clean_num(mean_source - overall_r if np.isfinite(mean_source) and np.isfinite(overall_r) else np.nan),
                "target_improvement": clean_num(mean_target - overall_r if np.isfinite(mean_target) and np.isfinite(overall_r) else np.nan),
                "joint_improvement": clean_num(mean_joint - overall_r if np.isfinite(mean_joint) and np.isfinite(overall_r) else np.nan),
            }
        )

    degree_strat_df = pd.DataFrame(degree_rows)
    degree_file = args.results_dir / "degree_stratified_correlations.csv"
    degree_strat_df.to_csv(degree_file, index=False)
    print(f"Saved: {degree_file}")

    # Linear correction analysis
    corr_rows: list[dict[str, float | str | None]] = []
    for name in analyzed_names:
        mp_data = residuals_df[residuals_df["metapath"] == name]
        if len(mp_data) < 3:
            continue
        x = mp_data["predicted"].to_numpy().reshape(-1, 1)
        y = mp_data["actual"].to_numpy()

        model = LinearRegression()
        model.fit(x, y)
        alpha = float(model.coef_[0])
        beta = float(model.intercept_)
        r2 = float(r2_score(y, model.predict(x))) if np.std(y) > 0 else np.nan

        original_r = safe_pearson(mp_data["predicted"].to_numpy(), y)
        corrected_pred = alpha * mp_data["predicted"].to_numpy() + beta
        corrected_r = safe_pearson(corrected_pred, y)

        corr_rows.append(
            {
                "metapath": name,
                "alpha": clean_num(alpha),
                "beta": clean_num(beta),
                "r2": clean_num(r2),
                "original_r": clean_num(original_r),
                "corrected_r": clean_num(corrected_r),
                "improvement": clean_num(
                    corrected_r - original_r
                    if np.isfinite(corrected_r) and np.isfinite(original_r)
                    else np.nan
                ),
            }
        )

    correction_df = pd.DataFrame(corr_rows)
    correction_file = args.results_dir / "correction_analysis.csv"
    correction_df.to_csv(correction_file, index=False)
    print(f"Saved: {correction_file}")

    if not args.skip_plot:
        maybe_write_plots(
            residuals_df=residuals_df,
            metapaths=metapaths,
            correction_df=correction_df,
            degree_strat_df=degree_strat_df,
            results_dir=args.results_dir,
            random_seed=args.random_seed,
        )
        print(f"Saved plots under: {args.results_dir / 'plots'}")

    print("\n" + "=" * 100)
    print("COMPOSITIONAL FAILURE ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"Saved: {csv_file}")
    print(f"Saved: {degree_file}")
    print(f"Saved: {correction_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
