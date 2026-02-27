"""Generate degree-preserving Hetionet permutations (no papermill required).

Behavior
- Requires an existing hetmat in ``data/`` (run `poe fetch-hetmat` first).
- Cleans any ``*.hetmat.bak`` leftovers.
- Skips work if the requested number of permutations already exists.
- Names permutations with zero-padded integers (000, 001, ...).

Usage
```
poe generate-permutations                      # default: generate up to 50 total
poe generate-permutations -- --count 10        # cap at 10 total permutations
poe generate-permutations -- --count 60 --seed 123
```
"""
import argparse
import itertools
import shutil
import sys
from pathlib import Path

import hetmatpy.hetmat
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PERM_DIR = DATA / "permutations"


def existing_permutation_ids():
    """Return sorted list of existing permutation integers."""
    if not PERM_DIR.exists():
        return []
    ids = []
    for d in PERM_DIR.iterdir():
        if d.is_dir() and d.name.endswith(".hetmat"):
            stem = d.name.split(".hetmat")[0]
            if stem.isdigit():
                ids.append(int(stem))
    return sorted(ids)


def clean_bak_dirs():
    """Remove any leftover *.hetmat.bak directories."""
    if not PERM_DIR.exists():
        return
    for bak in PERM_DIR.glob("*.hetmat.bak"):
        print(f"Removing stale backup: {bak.name}")
        shutil.rmtree(bak, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate degree-preserving Hetionet permutations."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Total permutations desired (including any that already exist).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; incremented per permutation.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional starting permutation index (default = next unused).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    edges_dir = DATA / "edges"
    if not edges_dir.exists():
        print("ERROR: data/edges missing. Run `poe fetch-hetmat` first.", file=sys.stderr)
        sys.exit(1)

    clean_bak_dirs()

    existing = existing_permutation_ids()
    if len(existing) >= args.count:
        print(f"{len(existing)} permutations already present; target {args.count}. Nothing to do.")
        return

    start_idx = args.start if args.start is not None else (max(existing) + 1 if existing else 0)
    num_new = args.count - len(existing)

    print(
        f"Generating {num_new} permutation(s) starting at {start_idx:03d} "
        f"(seed base {args.seed})"
    )

    hetmat = hetmatpy.hetmat.HetMat(DATA)
    namer = (f"{x:03d}" for x in itertools.count(start=start_idx))

    all_stats = []
    for i in range(num_new):
        perm_name = next(namer)
        # Advance seed per permutation for reproducibility differences
        perm_seed = args.seed + i
        print(f"[{i+1}/{num_new}] starting permutation {perm_name} (seed {perm_seed})", flush=True)
        perm_stats = hetmat.permute_graph(
            num_new_permutations=1,
            namer=iter([perm_name]),
            seed=perm_seed,
        )
        metaedge_count = perm_stats["metaedge"].nunique()
        print(f"[{i+1}/{num_new}] built permutation {perm_name} (seed {perm_seed}) "
              f"with {metaedge_count} metaedges")
        all_stats.append(perm_stats)

    stats = pd.concat(all_stats, ignore_index=True)

    # Summarize results
    by_perm = stats.groupby("permutation")["metaedge"].nunique().reset_index()
    print("\nPermutation summary (metaedges per perm):")
    for _, row in by_perm.iterrows():
        print(f"  {row['permutation']}: {row['metaedge']} metaedges")

    # Save stats for reference
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = PERM_DIR / "permutation_stats.csv"
    mode = "a" if stats_path.exists() else "w"
    header = not stats_path.exists()
    stats.to_csv(stats_path, mode=mode, header=header, index=False)
    print(f"\nSaved detailed stats to {stats_path}")


if __name__ == "__main__":
    main()
