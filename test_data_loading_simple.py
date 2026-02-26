"""
Simple test to verify data_loading module works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dwpc_pvalue_validation import data_loading

def main():
    print("Testing data_loading module...")
    print("=" * 60)

    # Get loader
    loader = data_loading.get_loader()
    print("✓ Created loader")

    # Test loading true Hetionet
    print("\n1. Loading true Hetionet HetMat...")
    hetmat_true = loader.load_hetmat("true")
    print(f"   ✓ Loaded: {hetmat_true}")
    print(f"   Metagraph: {hetmat_true.metagraph}")

    # Test loading a specific edge matrix
    print("\n2. Loading Compound-binds-Gene edge matrix...")
    matrix = loader.load_edge_matrix("CbG", source="true")
    print(f"   ✓ Shape: {matrix.shape}")
    print(f"   ✓ Edges: {matrix.nnz}")

    # Test getting degrees
    print("\n3. Getting node degrees...")
    source_degrees = loader.get_node_degrees("CbG", source="true", node_position="source")
    target_degrees = loader.get_node_degrees("CbG", source="true", node_position="target")
    print(f"   ✓ Source (Compound) degrees: min={source_degrees.min()}, max={source_degrees.max()}, mean={source_degrees.mean():.2f}")
    print(f"   ✓ Target (Gene) degrees: min={target_degrees.min()}, max={target_degrees.max()}, mean={target_degrees.mean():.2f}")

    # Test loading permutation 0
    print("\n4. Loading permutation 0...")
    hetmat_perm0 = loader.load_hetmat(perm_idx=0)
    print(f"   ✓ Loaded: {hetmat_perm0}")

    # Test loading permutation 1
    print("\n5. Loading permutation 1...")
    matrix_perm1 = loader.load_edge_matrix("CbG", perm_idx=1)
    print(f"   ✓ Shape: {matrix_perm1.shape}")
    print(f"   ✓ Edges: {matrix_perm1.nnz}")

    # Verify degree preservation
    print("\n6. Verifying degree preservation...")
    perm_source_degrees = loader.get_node_degrees("CbG", source="perm1", node_position="source")
    perm_target_degrees = loader.get_node_degrees("CbG", source="perm1", node_position="target")

    assert (source_degrees == perm_source_degrees).all(), "Source degrees should be preserved"
    assert (target_degrees == perm_target_degrees).all(), "Target degrees should be preserved"
    print(f"   ✓ Degrees are preserved in permutation")

    # Test metapath parsing
    print("\n7. Testing metapath parsing...")
    metaedges = data_loading.get_metaedges_for_metapath("CbGpPW")
    print(f"   CbGpPW -> {metaedges}")

    metaedges2 = data_loading.get_metaedges_for_metapath("CbGpPWpG")
    print(f"   CbGpPWpG -> {metaedges2}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
