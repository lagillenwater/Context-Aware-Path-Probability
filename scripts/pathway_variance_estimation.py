"""Estimate pathway-count variance across permutations (notebook 18g migration)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr

REPO_DIR = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(REPO_DIR / ".cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(REPO_DIR / ".cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

import matplotlib.pyplot as plt

from src.intermediate_signatures import (
    assign_to_bins,
    compute_intermediate_signature,
    create_degree_bins,
    extract_training_features,
)
from src.pathway_model_io import load_degree_sig_nn, predict_degree_sig_nn

DEFAULT_METAPATHS: dict[str, tuple[str, str]] = {
    "CbGpPW": ("CbG", "GpPW"),
    "CtDaG": ("CtD", "DaG"),
    "CbGaD": ("CbG", "GaD"),
    "CrCbG": ("CrC", "CbG"),
    "CbGiG": ("CbG", "GiG"),
    "CpDaG": ("CpD", "DaG"),
    "CbGpBP": ("CbG", "GpBP"),
    "CbGpCC": ("CbG", "GpCC"),
}


@dataclass(frozen=True)
class MetapathSpec:
    metapath: str
    edge1_type: str
    edge2_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate trained Degree Sig NN models on permutations and estimate variance. "
            "Script-first replacement for notebook 18g."
        )
    )
    parser.add_argument(
        "--metapath",
        action="append",
        default=[],
        help=(
            "Metapath key to process (repeatable). "
            f"Supported defaults: {', '.join(DEFAULT_METAPATHS.keys())}"
        ),
    )
    parser.add_argument(
        "--all-metapaths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process all default metapaths.",
    )
    parser.add_argument("--edge1-type", default=None, help="Override edge1 type for custom metapath run.")
    parser.add_argument("--edge2-type", default=None, help="Override edge2 type for custom metapath run.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Data directory containing permutations/ and edges/.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "trained_models",
        help="Directory containing <metapath>_Degree_Sig_NN.pt checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "variance_analysis",
        help="Output directory for variance-estimation artifacts.",
    )
    parser.add_argument("--n-permutations", type=int, default=20, help="Number of permutations to evaluate.")
    parser.add_argument("--first-perm-id", type=int, default=1, help="Starting permutation id.")
    parser.add_argument("--n-degree-bins", type=int, default=10, help="Degree bin count.")
    parser.add_argument("--n-inter-bins", type=int, default=10, help="Intermediate degree bin count.")
    parser.add_argument("--predict-batch-size", type=int, default=4096, help="Batch size for model predictions.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Torch device used for prediction.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (recorded in metadata).")
    parser.add_argument(
        "--skip-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip permutation validation plot generation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue to remaining metapaths if one fails.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.n_permutations <= 0:
        raise ValueError("--n-permutations must be > 0")
    if args.first_perm_id <= 0:
        raise ValueError("--first-perm-id must be > 0")
    if args.n_degree_bins <= 0:
        raise ValueError("--n-degree-bins must be > 0")
    if args.n_inter_bins <= 0:
        raise ValueError("--n-inter-bins must be > 0")
    if args.predict_batch_size <= 0:
        raise ValueError("--predict-batch-size must be > 0")
    if (args.edge1_type is None) ^ (args.edge2_type is None):
        raise ValueError("Provide both --edge1-type and --edge2-type together.")
    if args.edge1_type is not None and args.all_metapaths:
        raise ValueError("--edge1-type/--edge2-type cannot be combined with --all-metapaths.")
    if args.edge1_type is not None and len(args.metapath) > 1:
        raise ValueError("--edge1-type/--edge2-type supports one metapath label at a time.")
    unknown = [name for name in args.metapath if name not in DEFAULT_METAPATHS]
    if unknown and args.edge1_type is None:
        raise ValueError(
            f"Unknown --metapath values: {', '.join(unknown)}. "
            f"Supported: {', '.join(DEFAULT_METAPATHS.keys())}"
        )


def resolve_metapaths(args: argparse.Namespace) -> list[MetapathSpec]:
    if args.edge1_type is not None:
        label = args.metapath[0] if args.metapath else "custom"
        return [MetapathSpec(metapath=label, edge1_type=args.edge1_type, edge2_type=args.edge2_type)]

    if args.all_metapaths:
        names = list(DEFAULT_METAPATHS.keys())
    elif args.metapath:
        names = args.metapath
    else:
        names = ["CbGpPW"]

    return [
        MetapathSpec(metapath=name, edge1_type=DEFAULT_METAPATHS[name][0], edge2_type=DEFAULT_METAPATHS[name][1])
        for name in names
    ]


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
    return device_arg


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return float("nan")


def run_one_metapath(spec: MetapathSpec, args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    model_file = args.model_dir / f"{spec.metapath}_Degree_Sig_NN.pt"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_file}. "
            "Run pathway-train-degree-signature-nn first."
        )

    model, model_metadata = load_degree_sig_nn(model_file, device=device)
    print(f"Loaded model: {model_file}")
    expected_input_dim = int(getattr(model, "input_dim", 0))
    configured_input_dim = 2 + (args.n_inter_bins * args.n_inter_bins)
    runtime_inter_bins = args.n_inter_bins
    if expected_input_dim and expected_input_dim != configured_input_dim:
        inferred_side = int(round((expected_input_dim - 2) ** 0.5))
        if inferred_side > 0 and (2 + inferred_side * inferred_side) == expected_input_dim:
            runtime_inter_bins = inferred_side
            print(
                "Model/input mismatch detected; "
                f"using n_inter_bins={runtime_inter_bins} from checkpoint input_dim={expected_input_dim}."
            )
        else:
            raise ValueError(
                "Model input dimension does not match configured n_inter_bins and cannot be inferred: "
                f"expected_input_dim={expected_input_dim}, configured_input_dim={configured_input_dim}"
            )

    permutation_results: list[dict[str, Any]] = []

    for perm_offset in range(args.n_permutations):
        perm_id = args.first_perm_id + perm_offset
        perm_dir = args.data_dir / "permutations" / f"{perm_id:03d}.hetmat" / "edges"
        edge1_file = perm_dir / f"{spec.edge1_type}.sparse.npz"
        edge2_file = perm_dir / f"{spec.edge2_type}.sparse.npz"

        if not edge1_file.exists() or not edge2_file.exists():
            print(f"Warning: missing permutation {perm_id:03d}, skipping")
            continue

        edge1_matrix = sp.load_npz(str(edge1_file))
        edge2_matrix = sp.load_npz(str(edge2_file))

        if edge1_matrix.dtype == bool or edge1_matrix.dtype == np.bool_:
            edge1_matrix = edge1_matrix.astype(np.int32)
        if edge2_matrix.dtype == bool or edge2_matrix.dtype == np.bool_:
            edge2_matrix = edge2_matrix.astype(np.int32)

        pathway_matrix = edge1_matrix @ edge2_matrix
        if pathway_matrix.dtype == bool or pathway_matrix.dtype == np.bool_:
            pathway_matrix = pathway_matrix.astype(np.int32)

        source_degrees = np.asarray(edge1_matrix.sum(axis=1)).ravel()
        target_degrees = np.asarray(edge2_matrix.sum(axis=1)).ravel()

        source_bins = create_degree_bins(source_degrees, args.n_degree_bins)
        target_bins = create_degree_bins(target_degrees, args.n_degree_bins)

        signatures = compute_intermediate_signature(
            edge1_matrix=edge1_matrix,
            edge2_matrix=edge2_matrix,
            source_degrees=source_degrees,
            target_degrees=target_degrees,
            source_bins=source_bins,
            target_bins=target_bins,
            n_intermediate_bins=runtime_inter_bins,
        )

        X_signatures, bin_pairs = extract_training_features(signatures, normalize=True)
        X_perm = np.hstack([bin_pairs.astype(np.float32), X_signatures.astype(np.float32)])

        pathway_coo = pathway_matrix.tocoo()
        pathway_dict = {(i, j): v for i, j, v in zip(pathway_coo.row, pathway_coo.col, pathway_coo.data)}

        source_bin_assignments = assign_to_bins(source_degrees, source_bins)
        target_bin_assignments = assign_to_bins(target_degrees, target_bins)

        bin_pathway_counts: dict[tuple[int, int], list[float]] = {}
        for (i, j), count in pathway_dict.items():
            src_bin = int(source_bin_assignments[i])
            tgt_bin = int(target_bin_assignments[j])
            key = (src_bin, tgt_bin)
            if key not in bin_pathway_counts:
                bin_pathway_counts[key] = []
            bin_pathway_counts[key].append(float(count))

        y_perm_actual = np.array(
            [np.mean(bin_pathway_counts.get((int(src_bin), int(tgt_bin)), [0.0])) for src_bin, tgt_bin in bin_pairs],
            dtype=float,
        )
        y_perm_predicted = predict_degree_sig_nn(
            model,
            X_perm,
            device=device,
            batch_size=args.predict_batch_size,
        ).astype(float)

        r = safe_pearsonr(y_perm_predicted, y_perm_actual)
        mae = float(np.mean(np.abs(y_perm_predicted - y_perm_actual)))
        rmse = float(np.sqrt(np.mean((y_perm_predicted - y_perm_actual) ** 2)))

        print(f"Permutation {perm_id:03d}: r={r:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        permutation_results.append(
            {
                "perm_id": perm_id,
                "correlation": r,
                "mae": mae,
                "rmse": rmse,
                "predictions": y_perm_predicted,
                "actual": y_perm_actual,
                "bin_pairs": bin_pairs.copy(),
            }
        )

    if not permutation_results:
        raise RuntimeError("No permutation results were produced. Check permutation inputs.")

    correlations = [r["correlation"] for r in permutation_results]
    maes = [r["mae"] for r in permutation_results]
    rmses = [r["rmse"] for r in permutation_results]

    all_bin_pairs = permutation_results[0]["bin_pairs"]
    variance_estimates: list[dict[str, Any]] = []
    for bin_idx, (src_bin, tgt_bin) in enumerate(all_bin_pairs):
        counts_across_perms = [res["actual"][bin_idx] for res in permutation_results]
        variance_estimates.append(
            {
                "source_bin": int(src_bin),
                "target_bin": int(tgt_bin),
                "mean_count_across_perms": float(np.mean(counts_across_perms)),
                "std_count_across_perms": float(np.std(counts_across_perms)),
                "median_count": float(np.median(counts_across_perms)),
                "q25": float(np.percentile(counts_across_perms, 25)),
                "q75": float(np.percentile(counts_across_perms, 75)),
                "ci_lower_95": float(np.percentile(counts_across_perms, 2.5)),
                "ci_upper_95": float(np.percentile(counts_across_perms, 97.5)),
                "model_prediction": float(permutation_results[0]["predictions"][bin_idx]),
                "n_permutations": int(len(permutation_results)),
            }
        )

    variance_df = pd.DataFrame(variance_estimates)
    metrics_df = pd.DataFrame(
        {
            "permutation_id": [r["perm_id"] for r in permutation_results],
            "correlation": correlations,
            "mae": maes,
            "rmse": rmses,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_file: Path | None = None
    if not args.skip_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].bar(range(1, len(permutation_results) + 1), correlations, color="steelblue", alpha=0.7)
        axes[0, 0].axhline(np.nanmean(correlations), color="red", linestyle="--", label=f"Mean={np.nanmean(correlations):.4f}")
        axes[0, 0].set_xlabel("Permutation index")
        axes[0, 0].set_ylabel("Pearson correlation")
        axes[0, 0].set_title("Model Performance Across Permutations")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].bar(range(1, len(permutation_results) + 1), maes, color="coral", alpha=0.7)
        axes[0, 1].axhline(np.mean(maes), color="red", linestyle="--", label=f"Mean={np.mean(maes):.4f}")
        axes[0, 1].set_xlabel("Permutation index")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].set_title("MAE Across Permutations")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        perm0 = permutation_results[0]
        axes[1, 0].scatter(perm0["actual"], perm0["predictions"], alpha=0.6, s=30)
        lo = float(min(np.min(perm0["actual"]), np.min(perm0["predictions"])))
        hi = float(max(np.max(perm0["actual"]), np.max(perm0["predictions"])))
        axes[1, 0].plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
        axes[1, 0].set_xlabel("Actual count (first permutation)")
        axes[1, 0].set_ylabel("Predicted count")
        axes[1, 0].set_title(f"First permutation: r = {perm0['correlation']:.4f}")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        mean_std = float(variance_df["std_count_across_perms"].mean())
        axes[1, 1].hist(variance_df["std_count_across_perms"], bins=30, color="mediumpurple", alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(mean_std, color="red", linestyle="--", label=f"Mean={mean_std:.2f}")
        axes[1, 1].set_xlabel("Std dev across permutations")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Variance Estimates")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = args.output_dir / f"{spec.metapath}_permutation_validation.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    for perm_result in permutation_results:
        pred_file = args.output_dir / f"permutation_{int(perm_result['perm_id']):03d}_predictions.npy"
        np.save(pred_file, perm_result["predictions"])

    aggregated_file = args.output_dir / "all_permutations_results.npz"
    np.savez_compressed(
        aggregated_file,
        correlations=np.array(correlations),
        maes=np.array(maes),
        rmses=np.array(rmses),
        perm_ids=np.array([r["perm_id"] for r in permutation_results]),
    )

    variance_file = args.output_dir / f"{spec.metapath}_variance_estimates.csv"
    variance_df.to_csv(variance_file, index=False)

    metrics_file = args.output_dir / f"{spec.metapath}_permutation_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    summary = {
        "metapath": spec.metapath,
        "edge1_type": spec.edge1_type,
        "edge2_type": spec.edge2_type,
        "n_permutations": int(len(permutation_results)),
        "mean_correlation": float(np.nanmean(correlations)),
        "std_correlation": float(np.nanstd(correlations)),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "n_bin_combinations": int(len(variance_df)),
        "mean_variance": float(variance_df["std_count_across_perms"].mean()),
        "model_file": str(model_file),
        "model_metadata": model_metadata,
    }
    summary_file = args.output_dir / f"{spec.metapath}_validation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    return {
        "metapath": spec.metapath,
        "variance_file": str(variance_file),
        "metrics_file": str(metrics_file),
        "summary_file": str(summary_file),
        "plot_file": str(plot_file) if plot_file else None,
        "n_permutations_processed": int(len(permutation_results)),
        "mean_correlation": float(np.nanmean(correlations)),
    }


def args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, list):
            out[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)
    device = select_device(args.device)
    specs = resolve_metapaths(args)

    print(f"Data dir: {args.data_dir}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Metapaths: {', '.join(spec.metapath for spec in specs)}")

    successful: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for spec in specs:
        print("=" * 80)
        print(f"VARIANCE ESTIMATION: {spec.metapath} ({spec.edge1_type} -> {spec.edge2_type})")
        print("=" * 80)
        try:
            result = run_one_metapath(spec, args, device=device)
            successful.append(result)
            print(
                f"Complete {spec.metapath}: perms={result['n_permutations_processed']}, "
                f"mean r={result['mean_correlation']:.4f}"
            )
            print(f"Saved variance: {result['variance_file']}")
            print(f"Saved metrics: {result['metrics_file']}")
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            failed.append({"metapath": spec.metapath, "error": message})
            print(f"FAILED {spec.metapath}: {message}")
            if not args.continue_on_error:
                raise

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_file = args.output_dir / "pathway_variance_estimation_run_summary.json"
    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": args_to_json(args),
        "device": device,
        "successful_metapaths": [x["metapath"] for x in successful],
        "failed_metapaths": failed,
        "n_successful": len(successful),
        "n_failed": len(failed),
    }
    run_summary_file.write_text(json.dumps(run_summary, indent=2))
    print(f"Run summary: {run_summary_file}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
