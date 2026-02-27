"""Validate compositional metapath predictions without papermill.

This script ports notebook 17 (compositional validation) into a direct CLI.
It validates 2-hop metapath predictions against held-out permutations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


REPO_DIR = Path(__file__).resolve().parents[1]


@dataclass
class MetapathSpec:
    name: str
    edge1: str
    edge2: str
    description: str
    transpose_edge1: bool = False
    transpose_edge2: bool = False


DEFAULT_METAPATHS: list[MetapathSpec] = [
    MetapathSpec(
        name="CbGpPW",
        edge1="CbG",
        edge2="GpPW",
        description="Compound-binds-Gene-participates-Pathway",
    ),
    MetapathSpec(
        name="CtDaG",
        edge1="CtD",
        edge2="DaG",
        description="Compound-treats-Disease-associates-Gene",
    ),
    MetapathSpec(
        name="CbGaD",
        edge1="CbG",
        edge2="DaG",
        description="Compound-binds-Gene-associates-Disease",
        transpose_edge2=True,
    ),
    MetapathSpec(
        name="CrCbG",
        edge1="CrC",
        edge2="CbG",
        description="Compound-resembles-Compound-binds-Gene",
    ),
    MetapathSpec(
        name="CbGiG",
        edge1="CbG",
        edge2="GiG",
        description="Compound-binds-Gene-interacts-Gene",
    ),
    MetapathSpec(
        name="CpDaG",
        edge1="CpD",
        edge2="DaG",
        description="Compound-palliates-Disease-associates-Gene",
    ),
    MetapathSpec(
        name="CbGpBP",
        edge1="CbG",
        edge2="GpBP",
        description="Compound-binds-Gene-participates-BiologicalProcess",
    ),
    MetapathSpec(
        name="CbGpCC",
        edge1="CbG",
        edge2="GpCC",
        description="Compound-binds-Gene-participates-CellularComponent",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate compositional metapath calculation using training "
            "permutations for edge probabilities and held-out permutations "
            "for pathway-count comparisons."
        )
    )
    parser.add_argument(
        "--train-perms-start",
        type=int,
        default=1,
        help="First training permutation (inclusive). Default: 1.",
    )
    parser.add_argument(
        "--train-perms-end",
        type=int,
        default=20,
        help="Last training permutation (inclusive). Default: 20.",
    )
    parser.add_argument(
        "--valid-perms-start",
        type=int,
        default=21,
        help="First validation permutation (inclusive). Default: 21.",
    )
    parser.add_argument(
        "--valid-perms-end",
        type=int,
        default=30,
        help="Last validation permutation (inclusive). Default: 30.",
    )
    parser.add_argument(
        "--metapath",
        action="append",
        default=[],
        help=(
            "Default metapath name to run (repeatable). "
            f"Available: {', '.join(mp.name for mp in DEFAULT_METAPATHS)}"
        ),
    )
    parser.add_argument(
        "--permutations-dir",
        type=Path,
        default=REPO_DIR / "data" / "permutations",
        help="Directory containing ###.hetmat permutation folders.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "compositional_validation",
        help="Output directory for validation artifacts.",
    )
    parser.add_argument(
        "--max-compared-pairs",
        type=int,
        default=None,
        help="Optional cap on sparse pair comparisons per permutation for smoke tests.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for optional pair subsampling.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip output plots under results/compositional_validation/plots.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue to next metapath if one fails. Default: true.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.train_perms_start > args.train_perms_end:
        raise ValueError("train-perms-start must be <= train-perms-end")
    if args.valid_perms_start > args.valid_perms_end:
        raise ValueError("valid-perms-start must be <= valid-perms-end")
    if args.max_compared_pairs is not None and args.max_compared_pairs <= 0:
        raise ValueError("max-compared-pairs must be > 0")


def resolve_metapaths(selected_names: list[str]) -> list[MetapathSpec]:
    by_name = {m.name: m for m in DEFAULT_METAPATHS}
    if not selected_names:
        return DEFAULT_METAPATHS

    resolved: list[MetapathSpec] = []
    for name in selected_names:
        if name not in by_name:
            raise ValueError(
                f"Unknown metapath '{name}'. "
                f"Choose from: {', '.join(by_name.keys())}"
            )
        resolved.append(by_name[name])
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
        raise FileNotFoundError(f"Edge file not found: {edge_file}")
    matrix = sp.load_npz(str(edge_file)).tocsr()
    return matrix.transpose().tocsr() if transpose else matrix


def compute_empirical_edge_probabilities(
    *,
    permutations_dir: Path,
    edge_type: str,
    perm_ids: list[int],
    transpose: bool = False,
) -> sp.csr_matrix:
    print(
        f"Computing empirical probabilities for {edge_type}"
        f"{' (T)' if transpose else ''} from permutations {perm_ids[0]}-{perm_ids[-1]}"
    )
    base = load_edge_matrix(
        permutations_dir=permutations_dir,
        perm_id=perm_ids[0],
        edge_type=edge_type,
        transpose=transpose,
    )
    edge_sum = sp.csr_matrix(base.shape, dtype=np.float64)
    for perm_id in perm_ids:
        edge_sum = edge_sum + load_edge_matrix(
            permutations_dir=permutations_dir,
            perm_id=perm_id,
            edge_type=edge_type,
            transpose=transpose,
        ).astype(np.float64)

    edge_probs = edge_sum / len(perm_ids)
    print(f"  Shape: {edge_probs.shape}, non-zero: {edge_probs.nnz:,}")
    return edge_probs.tocsr()


def compute_actual_metapath_matrix(
    *,
    permutations_dir: Path,
    perm_id: int,
    spec: MetapathSpec,
) -> sp.csr_matrix:
    edge1 = load_edge_matrix(
        permutations_dir=permutations_dir,
        perm_id=perm_id,
        edge_type=spec.edge1,
        transpose=spec.transpose_edge1,
    )
    edge2 = load_edge_matrix(
        permutations_dir=permutations_dir,
        perm_id=perm_id,
        edge_type=spec.edge2,
        transpose=spec.transpose_edge2,
    )
    if edge1.shape[1] != edge2.shape[0]:
        raise ValueError(
            f"Shape mismatch for {spec.name} at perm {perm_id}: "
            f"{edge1.shape} then {edge2.shape}"
        )
    return (edge1 @ edge2).tocsr()


def safe_corrs(pred_vals: np.ndarray, actual_vals: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(pred_vals) < 2:
        return (np.nan, np.nan), (np.nan, np.nan)
    if np.std(pred_vals) == 0 or np.std(actual_vals) == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    try:
        pearson = pearsonr(pred_vals, actual_vals)
    except Exception:
        pearson = (np.nan, np.nan)
    try:
        spearman = spearmanr(pred_vals, actual_vals)
        spearman_tuple = (float(spearman.correlation), float(spearman.pvalue))
    except Exception:
        spearman_tuple = (np.nan, np.nan)
    return (float(pearson[0]), float(pearson[1])), spearman_tuple


def sparse_alignment_values(
    pred_sparse: sp.spmatrix,
    actual_sparse: sp.spmatrix,
    max_compared_pairs: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    pred_coo = pred_sparse.tocoo()
    actual_coo = actual_sparse.tocoo()

    pred_dict = {(int(i), int(j)): float(v) for i, j, v in zip(pred_coo.row, pred_coo.col, pred_coo.data)}
    actual_dict = {(int(i), int(j)): float(v) for i, j, v in zip(actual_coo.row, actual_coo.col, actual_coo.data)}
    all_locs = list(set(pred_dict.keys()) | set(actual_dict.keys()))

    if max_compared_pairs is not None and len(all_locs) > max_compared_pairs:
        idx = rng.choice(len(all_locs), size=max_compared_pairs, replace=False)
        all_locs = [all_locs[int(i)] for i in idx]

    pred_vals = np.fromiter((pred_dict.get(loc, 0.0) for loc in all_locs), dtype=np.float64, count=len(all_locs))
    actual_vals = np.fromiter((actual_dict.get(loc, 0.0) for loc in all_locs), dtype=np.float64, count=len(all_locs))
    return pred_vals, actual_vals, len(all_locs)


def validate_metapath(
    *,
    spec: MetapathSpec,
    train_perm_ids: list[int],
    valid_perm_ids: list[int],
    permutations_dir: Path,
    max_compared_pairs: int | None,
    rng: np.random.Generator,
    edge_prob_cache: dict[tuple[str, bool], sp.csr_matrix],
) -> dict[str, object]:
    print("\n" + "=" * 70)
    print(f"VALIDATING {spec.name}: {spec.edge1} -> {spec.edge2}")
    print(spec.description)
    print("=" * 70)

    key1 = (spec.edge1, spec.transpose_edge1)
    key2 = (spec.edge2, spec.transpose_edge2)
    if key1 not in edge_prob_cache:
        edge_prob_cache[key1] = compute_empirical_edge_probabilities(
            permutations_dir=permutations_dir,
            edge_type=spec.edge1,
            perm_ids=train_perm_ids,
            transpose=spec.transpose_edge1,
        )
    if key2 not in edge_prob_cache:
        edge_prob_cache[key2] = compute_empirical_edge_probabilities(
            permutations_dir=permutations_dir,
            edge_type=spec.edge2,
            perm_ids=train_perm_ids,
            transpose=spec.transpose_edge2,
        )

    edge1_probs = edge_prob_cache[key1]
    edge2_probs = edge_prob_cache[key2]
    if edge1_probs.shape[1] != edge2_probs.shape[0]:
        raise ValueError(
            f"Predicted metapath shape mismatch for {spec.name}: "
            f"{edge1_probs.shape} then {edge2_probs.shape}"
        )

    predicted = (edge1_probs @ edge2_probs).tocsr()
    print(f"Predicted matrix shape: {predicted.shape}, non-zero: {predicted.nnz:,}")

    per_perm_rows: list[dict[str, float | int | str]] = []
    for perm_id in valid_perm_ids:
        actual = compute_actual_metapath_matrix(
            permutations_dir=permutations_dir,
            perm_id=perm_id,
            spec=spec,
        )

        pred_vals, actual_vals, n_compared = sparse_alignment_values(
            predicted,
            actual,
            max_compared_pairs=max_compared_pairs,
            rng=rng,
        )
        (pearson_r, pearson_p), (spearman_r, spearman_p) = safe_corrs(pred_vals, actual_vals)

        if len(pred_vals) > 0:
            mae = float(mean_absolute_error(actual_vals, pred_vals))
            rmse = float(np.sqrt(mean_squared_error(actual_vals, pred_vals)))
            actual_mean = float(actual_vals.mean())
            predicted_mean = float(pred_vals.mean())
        else:
            mae = rmse = actual_mean = predicted_mean = np.nan

        per_perm_rows.append(
            {
                "perm_id": perm_id,
                "metapath": spec.name,
                "n_compared": int(n_compared),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "mae": mae,
                "rmse": rmse,
                "actual_mean": actual_mean,
                "predicted_mean": predicted_mean,
            }
        )
        print(
            f"  Perm {perm_id:03d}: r={pearson_r:.4f}, rho={spearman_r:.4f}, "
            f"MAE={mae:.4f}, RMSE={rmse:.4f} (n={n_compared:,})"
        )

    per_perm_df = pd.DataFrame(per_perm_rows)
    mean_pearson = float(per_perm_df["pearson_r"].mean())
    std_pearson = float(per_perm_df["pearson_r"].std())
    mean_spearman = float(per_perm_df["spearman_r"].mean())
    std_spearman = float(per_perm_df["spearman_r"].std())
    mean_mae = float(per_perm_df["mae"].mean())
    mean_rmse = float(per_perm_df["rmse"].mean())

    if mean_pearson > 0.95:
        decision = "PASS"
    elif mean_pearson > 0.85:
        decision = "BIAS"
    else:
        decision = "FAIL"

    print(f"Summary {spec.name}: mean r={mean_pearson:.4f}, decision={decision}")
    return {
        "metapath": spec.name,
        "edge1": spec.edge1,
        "edge2": spec.edge2,
        "n_hops": 2,
        "mean_pearson_r": mean_pearson,
        "std_pearson_r": std_pearson,
        "mean_spearman_r": mean_spearman,
        "std_spearman_r": std_spearman,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "decision": decision,
        "per_perm_results": per_perm_df,
    }


def save_plots(
    *,
    summary_df: pd.DataFrame,
    all_results: list[dict[str, object]],
    results_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlation by metapath
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df))
    colors = [
        "green" if d == "PASS" else "orange" if d == "BIAS" else "red"
        for d in summary_df["decision"]
    ]
    ax.bar(
        x,
        summary_df["mean_pearson_r"],
        yerr=summary_df["std_pearson_r"],
        color=colors,
        alpha=0.75,
        capsize=5,
    )
    ax.axhline(0.95, color="green", linestyle="--", linewidth=2, label="Pass (r=0.95)")
    ax.axhline(0.85, color="orange", linestyle="--", linewidth=2, label="Bias (r=0.85)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["metapath"], rotation=45, ha="right")
    ax.set_ylabel("Mean Pearson Correlation")
    ax.set_xlabel("Metapath")
    ax.set_title("Compositional Validation Accuracy by Metapath")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_by_metapath.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Correlation distribution across validation perms
    fig, ax = plt.subplots(figsize=(12, 8))
    combined_df = pd.concat(
        [r["per_perm_results"] for r in all_results],
        ignore_index=True,
    )
    metapath_order = summary_df["metapath"].tolist()
    sns.boxplot(
        data=combined_df,
        x="metapath",
        y="pearson_r",
        order=metapath_order,
        ax=ax,
        palette="Set2",
    )
    ax.axhline(0.95, color="green", linestyle="--", linewidth=2, label="Pass (r=0.95)")
    ax.axhline(0.85, color="orange", linestyle="--", linewidth=2, label="Bias (r=0.85)")
    ax.set_xlabel("Metapath")
    ax.set_ylabel("Pearson Correlation (per permutation)")
    ax.set_title("Correlation Distribution Across Validation Permutations")
    ax.set_xticklabels(metapath_order, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. MAE and RMSE comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(summary_df))
    ax1.bar(x, summary_df["mean_mae"], color="steelblue", alpha=0.75)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["metapath"], rotation=45, ha="right")
    ax1.set_title("MAE by Metapath")
    ax1.set_ylabel("Mean Absolute Error")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, summary_df["mean_rmse"], color="coral", alpha=0.75)
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df["metapath"], rotation=45, ha="right")
    ax2.set_title("RMSE by Metapath")
    ax2.set_ylabel("Root Mean Squared Error")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / "error_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_perm_ids = list(range(args.train_perms_start, args.train_perms_end + 1))
    valid_perm_ids = list(range(args.valid_perms_start, args.valid_perms_end + 1))
    metapaths = resolve_metapaths(args.metapath)
    rng = np.random.default_rng(args.random_seed)

    print(f"Permutation directory: {args.permutations_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Training permutations: {train_perm_ids[0]}-{train_perm_ids[-1]}")
    print(f"Validation permutations: {valid_perm_ids[0]}-{valid_perm_ids[-1]}")
    print(f"Metapaths to validate: {len(metapaths)}")

    edge_prob_cache: dict[tuple[str, bool], sp.csr_matrix] = {}
    all_results: list[dict[str, object]] = []

    for spec in metapaths:
        try:
            result = validate_metapath(
                spec=spec,
                train_perm_ids=train_perm_ids,
                valid_perm_ids=valid_perm_ids,
                permutations_dir=args.permutations_dir,
                max_compared_pairs=args.max_compared_pairs,
                rng=rng,
                edge_prob_cache=edge_prob_cache,
            )
            all_results.append(result)
        except Exception as exc:
            if args.continue_on_error:
                print(f"WARNING: failed {spec.name}: {exc}")
                continue
            raise

    if not all_results:
        raise RuntimeError("No metapath validations succeeded.")

    summary_df = pd.DataFrame(
        [
            {
                "metapath": r["metapath"],
                "n_hops": r["n_hops"],
                "mean_pearson_r": r["mean_pearson_r"],
                "std_pearson_r": r["std_pearson_r"],
                "mean_spearman_r": r["mean_spearman_r"],
                "mean_mae": r["mean_mae"],
                "mean_rmse": r["mean_rmse"],
                "decision": r["decision"],
            }
            for r in all_results
        ]
    ).sort_values("mean_pearson_r", ascending=False)

    overall_mean_r = float(summary_df["mean_pearson_r"].mean())
    n_pass = int((summary_df["decision"] == "PASS").sum())
    n_bias = int((summary_df["decision"] == "BIAS").sum())
    n_fail = int((summary_df["decision"] == "FAIL").sum())

    if overall_mean_r > 0.95:
        overall_decision = "VALIDATED"
        recommendation = "Proceed to notebook 18"
    elif overall_mean_r > 0.85:
        overall_decision = "BIAS_DETECTED"
        recommendation = "Consider bias correction"
    else:
        overall_decision = "FAILED"
        recommendation = "Use direct empirical pathway counts"

    summary_df.to_csv(args.results_dir / "accuracy_by_metapath.csv", index=False)
    per_perm_df = pd.concat([r["per_perm_results"] for r in all_results], ignore_index=True)
    per_perm_df.to_csv(args.results_dir / "per_permutation_metrics.csv", index=False)

    validation_summary = {
        "overall_decision": overall_decision,
        "overall_mean_pearson_r": overall_mean_r,
        "n_metapaths_tested": len(all_results),
        "n_pass": n_pass,
        "n_bias": n_bias,
        "n_fail": n_fail,
        "train_permutations": f"{args.train_perms_start}-{args.train_perms_end}",
        "validation_permutations": f"{args.valid_perms_start}-{args.valid_perms_end}",
        "decision_criteria": {
            "pass_threshold": 0.95,
            "bias_threshold": 0.85,
            "fail_below": 0.85,
        },
        "metapath_results": summary_df.to_dict(orient="records"),
        "recommendation": recommendation,
    }
    (args.results_dir / "validation_summary.json").write_text(json.dumps(validation_summary, indent=2))

    if not args.skip_plot:
        save_plots(
            summary_df=summary_df,
            all_results=all_results,
            results_dir=args.results_dir,
        )

    print("\n" + "=" * 100)
    print("COMPOSITIONAL VALIDATION COMPLETE")
    print("=" * 100)
    print(f"Metapaths tested: {len(all_results)}")
    print(f"Overall mean Pearson r: {overall_mean_r:.4f}")
    print(f"Decision: {overall_decision}")
    print(f"Saved: {args.results_dir / 'accuracy_by_metapath.csv'}")
    print(f"Saved: {args.results_dir / 'per_permutation_metrics.csv'}")
    print(f"Saved: {args.results_dir / 'validation_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
