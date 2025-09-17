"""
Simple pytest tests for permutation validation.
"""

import hashlib
from pathlib import Path


def file_hash(filepath):
    """Get MD5 hash of a file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_permutation_dirs():
    """Get all permutation directories."""
    data_dir = Path(__file__).parent.parent / "data" / "permutations"
    return sorted(data_dir.glob("*.hetmat"))


def test_permutations_exist():
    """Test that permutation directories exist."""
    perm_dirs = get_permutation_dirs()
    assert len(perm_dirs) >= 2, f"Need at least 2 permutations. Found: {len(perm_dirs)}"


def test_ctd_files_are_different():
    """Test that CtD edge files are different across permutations."""
    perm_dirs = get_permutation_dirs()
    
    if len(perm_dirs) < 2:
        return  # Skip if not enough permutations
    
    hashes = {}
    
    for perm_dir in perm_dirs:
        edge_file = perm_dir / "edges" / "CtD.sparse.npz"
        
        if edge_file.exists():
            file_hash_val = file_hash(edge_file)
            
            if file_hash_val in hashes:
                assert False, f"IDENTICAL CtD files: {hashes[file_hash_val]} ≡ {perm_dir.name}"
            
            hashes[file_hash_val] = perm_dir.name
    
    assert len(hashes) >= 2, "Need at least 2 CtD files to compare"
    print(f"All {len(hashes)} CtD files are different!")


def test_file_structure():
    """Test that permutations have proper structure."""
    perm_dirs = get_permutation_dirs()
    
    for perm_dir in perm_dirs:
        assert (perm_dir / "edges").exists(), f"Missing edges/ in {perm_dir.name}"
        assert (perm_dir / "nodes").exists(), f"Missing nodes/ in {perm_dir.name}"


if __name__ == "__main__":
    # Quick manual test
    test_permutations_exist()
    test_ctd_files_are_different()
    test_file_structure()
    print("All tests passed!")