"""Minimal hetmat fetch/build runner without papermill.
Assumes the notebook logic can be replaced by existing data snapshot.
If data/edges already exists, it will skip.
"""
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SNAP = REPO / "archive" / "snapshots" / "data_2025-02-26"


def main():
    if (DATA / "edges").exists():
        print("data/edges exists; skipping restore")
        return
    if not SNAP.exists():
        print(f"No snapshot found at {SNAP}; aborting", file=sys.stderr)
        sys.exit(1)
    print(f"Restoring data snapshot from {SNAP} -> {DATA}")
    shutil.copytree(SNAP, DATA)
    print("Done.")


if __name__ == "__main__":
    main()
