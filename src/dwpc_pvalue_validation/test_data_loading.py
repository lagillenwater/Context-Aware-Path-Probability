"""
Quick test of data_loading module
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dwpc_pvalue_validation import data_loading, config

def test_basic_loading():
    """Test basic data loading functionality."""
    print("="*60)
    print("Testing data_loading module")
    print("="*60)

    # Get loader
    loader = data_loading.get_loader()

    # Test with CbG (Compound-binds-Gene)
    metaedge = "CbG"
    print(f"\nTesting with metaedge: {metaedge}")

    # Load true Hetionet
    print("\n1. Loading true Hetionet...")
    matrix_true = loader.load_edge_matrix(metaedge, source="true")
    print(f"   Shape: {matrix_true.shape}")
    print(f"   Edges: {matrix_true.nnz}")
    print(f"   Density: {matrix_true.nnz / (matrix_true.shape[0] * matrix_true.shape[1]):.6f}")

    # Load permutation 0
    print("\n2. Loading permutation 0...")
    matrix_perm0 = loader.load_edge_matrix(metaedge, source="perm0")
    print(f"   Shape: {matrix_perm0.shape}")
    print(f"   Edges: {matrix_perm0.nnz}")

    # Verify same number of edges
    assert matrix_perm0.nnz == matrix_true.nnz, "Permutation should have same number of edges"
    print(f"   PASSED: Same number of edges as true Hetionet")

    # Load permutation 1
    print("\n3. Loading permutation 1...")
    matrix_perm1 = loader.load_edge_matrix(metaedge, perm_idx=1)
    print(f"   Shape: {matrix_perm1.shape}")
    print(f"   Edges: {matrix_perm1.nnz}")

    # Get degrees
    print("\n4. Getting node degrees...")
    source_degrees_true = loader.get_node_degrees(metaedge, source="true", node_position="source")
    target_degrees_true = loader.get_node_degrees(metaedge, source="true", node_position="target")

    print(f"   Source degrees (Compounds): min={source_degrees_true.min()}, "
          f"max={source_degrees_true.max()}, mean={source_degrees_true.mean():.2f}")
    print(f"   Target degrees (Genes): min={target_degrees_true.min()}, "
          f"max={target_degrees_true.max()}, mean={target_degrees_true.mean():.2f}")

    # Check degree preservation
    source_degrees_perm = loader.get_node_degrees(metaedge, source="perm1", node_position="source")
    target_degrees_perm = loader.get_node_degrees(metaedge, source="perm1", node_position="target")

    assert (source_degrees_true == source_degrees_perm).all(), "Permutation should preserve source degrees"
    assert (target_degrees_true == target_degrees_perm).all(), "Permutation should preserve target degrees"
    print(f"   PASSED: Permutation preserves degrees")

    # Get connected pairs
    print("\n5. Getting connected node pairs...")
    source_nodes, target_nodes = loader.get_connected_node_pairs(metaedge, source="true")
    print(f"   Connected pairs: {len(source_nodes)}")

    # Test metapath parsing
    print("\n6. Testing metapath parsing...")
    test_metapaths = ["CbGpPW", "CtDaGiG", "CbGpPWpGaD"]
    for mp in test_metapaths:
        metaedges = data_loading.get_metaedges_for_metapath(mp)
        print(f"   {mp} -> {metaedges}")

    # Test data availability
    print("\n7. Testing data availability...")
    for metapath_abbrev, _, _ in config.METAPATHS[:3]:  # Test first 3
        available = data_loading.validate_data_availability(metapath_abbrev)
        status = "PASS" if available else "FAIL"
        print(f"   {status} {metapath_abbrev}: {available}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_basic_loading()
