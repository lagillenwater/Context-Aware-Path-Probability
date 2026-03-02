"""Smart hetmat fetch: restore snapshot if present; otherwise fail with clear guidance.
- If data/edges exists, no-op.
- Else if archive snapshot exists, restore it.
- Else fail with instructions (expected first-time users to supply hetmat or run notebook externally).
"""
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TARGET = DATA / "edges"
SNAP_ROOT = REPO / "archive" / "snapshots" / "data_2025-02-26"
SNAP = SNAP_ROOT / "edges"


def main():
    if TARGET.exists() and any(TARGET.iterdir()):
        print("data/edges exists; skipping fetch")
        return

    DATA.mkdir(parents=True, exist_ok=True)

    if SNAP.exists():
        print(f"Restoring hetmat edges from {SNAP_ROOT} -> {DATA}")
        # copy tree of edges, nodes, metagraph.json, permutations if present
        for item in SNAP_ROOT.iterdir():
            dest = DATA / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        print("Done.")
        return

    print("ERROR: No hetmat found and no snapshot available.", file=sys.stderr)
    print("Please provide hetmat files under data/ (edges/, nodes/, metagraph.json) or run the original 0_create-hetmat notebook manually.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
