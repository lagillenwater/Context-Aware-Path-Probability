"""Fetch Hetionet hetmat.

Order:
1) If data/edges exists, skip.
2) Otherwise download Hetionet v1.0 JSON and build hetmat into data/.
"""
import os
import pathlib
import shutil
import sys

import hetnetpy.readwrite
import hetmatpy.hetmat

URL = "https://github.com/dhimmel/hetionet/raw/76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/hetnet/json/hetionet-v1.0.json.bz2"
REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def validate_metagraph(hetmat):
    metagraph = hetmat.metagraph
    metanodes = set(str(n) for n in metagraph.get_nodes())
    metaedges = set(str(e) for e in metagraph.get_edges())
    if len(metanodes) != 11:
        print(f"Warning: expected 11 metanodes, found {len(metanodes)}", file=sys.stderr)
    if len(metaedges) != 24:
        print(f"Warning: expected 24 metaedges, found {len(metaedges)}", file=sys.stderr)

    required = [DATA / "metagraph.json", DATA / "nodes", DATA / "edges"]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"ERROR: missing required outputs: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Metanodes ({len(metanodes)}): {sorted(metanodes)}")
    print(f"Metaedges ({len(metaedges)}): {sorted(metaedges)}")


def download_and_build():
    print(f"Downloading Hetionet v1.0 from {URL}")
    graph = hetnetpy.readwrite.read_graph(URL)

    print(f"Writing hetmat to {DATA}")
    hetmat = hetmatpy.hetmat.hetmat_from_graph(graph, DATA)
    validate_metagraph(hetmat)
    print("Hetmat build complete.")


def main():
    edges_dir = DATA / "edges"
    if edges_dir.exists() and any(edges_dir.iterdir()):
        print("data/edges exists; validating existing hetmat...")
        hetmat = hetmatpy.hetmat.HetMat.from_path(DATA)
        validate_metagraph(hetmat)
        return

    DATA.mkdir(parents=True, exist_ok=True)
    print("No hetmat found; downloading and building...")
    download_and_build()


if __name__ == "__main__":
    main()
