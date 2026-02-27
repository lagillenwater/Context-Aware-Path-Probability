"""Prepare degree-binned pathway training data (notebook 18a migration)."""

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

REPO_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_DIR / ".cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from src.intermediate_signatures import (
    assign_to_bins,
    compute_intermediate_signature,
    create_degree_bins,
    extract_training_features,
    get_signature_stats,
)

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
            "Generate degree-binned pathway training data used by the notebook-18 "
            "model suite. This is a script-first replacement for notebook 18a."
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
        help="Process all default metapaths in one run (default: false).",
    )
    parser.add_argument("--edge1-type", default=None, help="Override edge1 type for a custom metapath run.")
    parser.add_argument("--edge2-type", default=None, help="Override edge2 type for a custom metapath run.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_DIR / "data",
        help="Repository data directory containing edges/ and permutations/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_DIR / "results" / "pathway_nn" / "training_data",
        help="Output directory for <metapath>_training_data.csv files.",
    )
    parser.add_argument("--n-degree-bins", type=int, default=10, help="Number of source/target degree bins.")
    parser.add_argument("--n-inter-bins", type=int, default=10, help="Number of intermediate in/out degree bins.")
    parser.add_argument(
        "--target-degree-axis",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Axis used to compute target degrees from edge2 (default: 1 for "
            "notebook-18a parity)."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Run seed recorded in metadata.")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue to the next metapath if one metapath fails.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.n_degree_bins <= 0:
        raise ValueError("--n-degree-bins must be > 0")
    if args.n_inter_bins <= 0:
        raise ValueError("--n-inter-bins must be > 0")
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


def load_edge_matrix(data_dir: Path, edge_type: str) -> tuple[sp.spmatrix, Path]:
    primary = data_dir / "edges" / f"{edge_type}.sparse.npz"
    fallback = data_dir / "permutations" / "000.hetmat" / "edges" / f"{edge_type}.sparse.npz"

    if primary.exists():
        return sp.load_npz(str(primary)), primary
    if fallback.exists():
        return sp.load_npz(str(fallback)), fallback
    raise FileNotFoundError(f"Edge file not found for {edge_type}: checked {primary} and {fallback}")


def degree_stats(degrees: np.ndarray) -> dict[str, float]:
    nonzero = degrees[degrees > 0]
    if len(nonzero) == 0:
        return {"min_nonzero": 0.0, "max": float(np.max(degrees) if len(degrees) else 0.0), "mean_nonzero": 0.0}
    return {
        "min_nonzero": float(np.min(nonzero)),
        "max": float(np.max(degrees)),
        "mean_nonzero": float(np.mean(nonzero)),
    }


def build_training_dataframe(
    *,
    edge1_matrix: sp.spmatrix,
    edge2_matrix: sp.spmatrix,
    source_degrees: np.ndarray,
    target_degrees: np.ndarray,
    n_degree_bins: int,
    n_inter_bins: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    source_bins = create_degree_bins(source_degrees, n_degree_bins)
    target_bins = create_degree_bins(target_degrees, n_degree_bins)

    signatures = compute_intermediate_signature(
        edge1_matrix=edge1_matrix,
        edge2_matrix=edge2_matrix,
        source_degrees=source_degrees,
        target_degrees=target_degrees,
        source_bins=source_bins,
        target_bins=target_bins,
        n_intermediate_bins=n_inter_bins,
    )
    signature_stats = get_signature_stats(signatures)

    pathway_matrix = edge1_matrix @ edge2_matrix
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

    X_signatures, bin_pairs = extract_training_features(signatures, normalize=True)

    training_data: list[dict[str, Any]] = []
    for idx, (src_bin, tgt_bin) in enumerate(bin_pairs):
        sig_features = X_signatures[idx]
        counts = bin_pathway_counts.get((int(src_bin), int(tgt_bin)), [0.0])

        row: dict[str, Any] = {
            "source_bin": int(src_bin),
            "target_bin": int(tgt_bin),
            "pathway_count_mean": float(np.mean(counts)),
            "pathway_count_std": float(np.std(counts)),
            "pathway_count_median": float(np.median(counts)),
            "pathway_count_q25": float(np.percentile(counts, 25)),
            "pathway_count_q75": float(np.percentile(counts, 75)),
            "n_pairs_in_bin": int(len(counts)),
        }
        for i, val in enumerate(sig_features):
            row[f"inter_sig_{i}"] = float(val)
        training_data.append(row)

    df = pd.DataFrame(training_data)
    matrix_summary = {
        "source_shape": list(edge1_matrix.shape),
        "source_nnz": int(edge1_matrix.nnz),
        "target_shape": list(edge2_matrix.shape),
        "target_nnz": int(edge2_matrix.nnz),
        "pathway_shape": list(pathway_matrix.shape),
        "pathway_nnz": int(pathway_matrix.nnz),
        "pathway_min": float(np.min(pathway_matrix.data)) if pathway_matrix.nnz > 0 else 0.0,
        "pathway_max": float(np.max(pathway_matrix.data)) if pathway_matrix.nnz > 0 else 0.0,
    }

    return df, signature_stats, matrix_summary


def prepare_one_metapath(spec: MetapathSpec, args: argparse.Namespace) -> dict[str, Any]:
    print("=" * 80)
    print(f"PREPARING PATHWAY TRAINING DATA: {spec.metapath}")
    print("=" * 80)
    print(f"Edges: {spec.edge1_type} -> {spec.edge2_type}")

    edge1_matrix, edge1_file = load_edge_matrix(args.data_dir, spec.edge1_type)
    edge2_matrix, edge2_file = load_edge_matrix(args.data_dir, spec.edge2_type)

    if edge1_matrix.dtype == bool or edge1_matrix.dtype == np.bool_:
        edge1_matrix = edge1_matrix.astype(np.int32)
    if edge2_matrix.dtype == bool or edge2_matrix.dtype == np.bool_:
        edge2_matrix = edge2_matrix.astype(np.int32)

    source_degrees = np.asarray(edge1_matrix.sum(axis=1)).ravel()
    target_degrees = np.asarray(edge2_matrix.sum(axis=args.target_degree_axis)).ravel()

    print(f"Loaded edge1: {edge1_file}")
    print(f"Loaded edge2: {edge2_file}")
    print(
        f"Source nodes: {len(source_degrees)} | range: "
        f"{degree_stats(source_degrees)['min_nonzero']:.0f} - {degree_stats(source_degrees)['max']:.0f}"
    )
    print(
        f"Target nodes: {len(target_degrees)} | range: "
        f"{degree_stats(target_degrees)['min_nonzero']:.0f} - {degree_stats(target_degrees)['max']:.0f}"
    )

    df, signature_stats, matrix_summary = build_training_dataframe(
        edge1_matrix=edge1_matrix,
        edge2_matrix=edge2_matrix,
        source_degrees=source_degrees,
        target_degrees=target_degrees,
        n_degree_bins=args.n_degree_bins,
        n_inter_bins=args.n_inter_bins,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.results_dir / f"{spec.metapath}_training_data.csv"
    summary_file = args.results_dir / f"{spec.metapath}_training_data_summary.json"

    df.to_csv(output_file, index=False)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metapath": spec.metapath,
        "edge1_type": spec.edge1_type,
        "edge2_type": spec.edge2_type,
        "edge_files": {"edge1": str(edge1_file), "edge2": str(edge2_file)},
        "params": {
            "n_degree_bins": args.n_degree_bins,
            "n_inter_bins": args.n_inter_bins,
            "target_degree_axis": args.target_degree_axis,
            "random_seed": args.random_seed,
        },
        "matrix_summary": matrix_summary,
        "source_degree_stats": degree_stats(source_degrees),
        "target_degree_stats": degree_stats(target_degrees),
        "signature_stats": signature_stats,
        "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
        "output_csv": str(output_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2))

    print(f"Saved: {output_file}")
    print(f"Summary: {summary_file}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return summary


def args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            payload[key] = value
    return payload


def main() -> None:
    args = parse_args()
    specs = resolve_metapaths(args)

    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Metapaths to run: {', '.join(spec.metapath for spec in specs)}")

    successful: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for spec in specs:
        try:
            summary = prepare_one_metapath(spec, args)
            successful.append(summary)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {spec.metapath}: {message}")
            failed.append({"metapath": spec.metapath, "error": message})
            if not args.continue_on_error:
                raise

    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_summary_file = args.results_dir / "pathway_data_preparation_run_summary.json"
    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": args_to_json(args),
        "successful_metapaths": [item["metapath"] for item in successful],
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
