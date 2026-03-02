"""Fetch or validate the Hetionet hetmat.

Logic:
1) If ``data/`` already contains a hetmat, validate it.
2) Otherwise download Hetionet v1.0 JSON and build a fresh hetmat into ``data/``.
"""
import pathlib
import sys

import hetnetpy.readwrite
import hetmatpy.hetmat

URL = "https://github.com/dhimmel/hetionet/raw/76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/hetnet/json/hetionet-v1.0.json.bz2"
REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def validate_hetmat(directory: pathlib.Path) -> None:
    """Validate that a hetmat on disk looks like Hetionet v1.0."""
    required = [directory / "metagraph.json", directory / "nodes", directory / "edges"]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"ERROR: missing required outputs: {missing}", file=sys.stderr)
        sys.exit(1)

    hetmat = hetmatpy.hetmat.HetMat(directory)
    metagraph = hetmat.metagraph
    metanodes = {str(n) for n in metagraph.get_nodes()}
    metaedges = {str(e) for e in metagraph.get_edges()}

    print("\n=== Hetmat validation ===")
    print(f"Location : {directory}")

    if len(metanodes) != 11:
        print(f"Warning: expected 11 metanodes, found {len(metanodes)}", file=sys.stderr)
    if len(metaedges) != 24:
        print(f"Warning: expected 24 metaedges, found {len(metaedges)}", file=sys.stderr)

    print(f"Metanodes: {len(metanodes)}/11")
    print("  " + ", ".join(sorted(metanodes)))
    print(f"Metaedges: {len(metaedges)}/24")
    print("  " + ", ".join(sorted(metaedges)))
    print("Status   : ok\n")


def download_and_build():
    print(f"Downloading Hetionet v1.0 JSON from {URL}")
    graph = hetnetpy.readwrite.read_graph(URL)

    print(f"Writing hetmat to {DATA}")
    hetmat = hetmatpy.hetmat.hetmat_from_graph(graph, DATA)
    validate_hetmat(DATA)
    print("Hetmat build complete.")


def main():
    edges_dir = DATA / "edges"
    if edges_dir.exists() and any(edges_dir.iterdir()):
        print("data/edges exists; validating existing hetmat...")
        validate_hetmat(DATA)
        return

    DATA.mkdir(parents=True, exist_ok=True)
    print("No hetmat found; downloading and building...")
    download_and_build()


if __name__ == "__main__":
    main()
