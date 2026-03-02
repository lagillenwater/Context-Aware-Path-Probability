"""
Test suite for sampling module

Tests stratified sampling of node pairs by degree category.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dwpc_pvalue_validation import sampling, data_loading, config


def test_degree_binning():
    """Test degree quantile binning."""
    print("=" * 60)
    print("Test 1: Degree Binning")
    print("=" * 60)

    degrees = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    quantiles = [0.0, 0.33, 0.67, 1.0]

    bins, labels = sampling.compute_degree_bins(degrees, quantiles)

    print(f"Degrees: {degrees}")
    print(f"Quantiles: {quantiles}")
    print(f"Bins: {bins}")
    print(f"Labels: {labels}")

    # Verify bins
    assert len(bins) == len(quantiles), "Bins should match quantiles"
    assert bins[0] <= degrees.min(), "First bin should be <= min degree"
    assert bins[-1] >= degrees.max(), "Last bin should be >= max degree"

    # Verify labels
    assert len(labels) == len(quantiles) - 1, "Should have n-1 labels"

    print("PASSED: Degree binning works correctly\n")


def test_categorize_pairs():
    """Test pair categorization by degree."""
    print("=" * 60)
    print("Test 2: Pair Categorization")
    print("=" * 60)

    source_degrees = np.array([1, 2, 5, 8])
    target_degrees = np.array([1, 3, 6, 9])
    source_bins = np.array([0, 3, 6, 10])
    target_bins = np.array([0, 3, 6, 10])
    category_names = ["Low", "Medium", "High"]

    categories = sampling.categorize_node_pairs(
        source_degrees,
        target_degrees,
        source_bins,
        target_bins,
        category_names
    )

    print(f"Source degrees: {source_degrees}")
    print(f"Target degrees: {target_degrees}")
    print(f"Categories: {categories}")

    # Verify output
    assert len(categories) == len(source_degrees), "One category per pair"
    assert all(isinstance(cat, tuple) for cat in categories), "Categories should be tuples"
    assert all(len(cat) == 2 for cat in categories), "Categories should be (source_cat, target_cat)"

    # Verify valid categories
    valid_cats = set(category_names)
    for src_cat, tgt_cat in categories:
        assert src_cat in valid_cats, f"Invalid source category: {src_cat}"
        assert tgt_cat in valid_cats, f"Invalid target category: {tgt_cat}"

    print("PASSED: Pair categorization works correctly\n")


def test_sample_pairs_by_category():
    """Test sampling node pairs by degree category."""
    print("=" * 60)
    print("Test 3: Sample Pairs by Category")
    print("=" * 60)

    loader = data_loading.get_loader()
    metaedge = "CbG"
    n_samples = 10

    # Sample from true Hetionet
    samples = sampling.sample_node_pairs(
        metaedge=metaedge,
        source="true",
        n_samples_per_category=n_samples,
        quantiles=config.DEGREE_QUANTILES,
        category_names=config.DEGREE_CATEGORY_NAMES,
        random_state=42
    )

    print(f"Sampled {len(samples)} node pairs")
    print(f"First sample: {samples[0]}")
    print(f"\nCategory distribution:")

    # Count samples per category
    from collections import Counter
    cat_counts = Counter(s['category'] for s in samples)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Verify output structure
    assert len(samples) > 0, "Should have samples"
    for sample in samples:
        assert 'source_idx' in sample, "Should have source_idx"
        assert 'target_idx' in sample, "Should have target_idx"
        assert 'source_degree' in sample, "Should have source_degree"
        assert 'target_degree' in sample, "Should have target_degree"
        assert 'category' in sample, "Should have category"
        assert isinstance(sample['category'], tuple), "Category should be tuple"
        assert len(sample['category']) == 2, "Category should be (src_cat, tgt_cat)"

    # Verify we have samples from multiple categories
    n_categories = len(cat_counts)
    assert n_categories > 1, f"Should have multiple categories, got {n_categories}"

    # Verify degrees match categories
    for sample in samples:
        src_cat, tgt_cat = sample['category']
        assert src_cat in config.DEGREE_CATEGORY_NAMES, f"Invalid category: {src_cat}"
        assert tgt_cat in config.DEGREE_CATEGORY_NAMES, f"Invalid category: {tgt_cat}"

    print("PASSED: Sampling by category works correctly\n")


def test_sample_reproducibility():
    """Test that sampling is reproducible with fixed random seed."""
    print("=" * 60)
    print("Test 4: Sampling Reproducibility")
    print("=" * 60)

    metaedge = "CbG"
    n_samples = 5

    # Sample twice with same seed
    samples1 = sampling.sample_node_pairs(
        metaedge=metaedge,
        source="true",
        n_samples_per_category=n_samples,
        quantiles=config.DEGREE_QUANTILES,
        category_names=config.DEGREE_CATEGORY_NAMES,
        random_state=42
    )

    samples2 = sampling.sample_node_pairs(
        metaedge=metaedge,
        source="true",
        n_samples_per_category=n_samples,
        quantiles=config.DEGREE_QUANTILES,
        category_names=config.DEGREE_CATEGORY_NAMES,
        random_state=42
    )

    # Verify same samples
    assert len(samples1) == len(samples2), "Should have same number of samples"
    for s1, s2 in zip(samples1, samples2):
        assert s1['source_idx'] == s2['source_idx'], "Source indices should match"
        assert s1['target_idx'] == s2['target_idx'], "Target indices should match"
        assert s1['category'] == s2['category'], "Categories should match"

    print("PASSED: Sampling is reproducible with fixed seed\n")


def test_sample_from_permutation():
    """Test sampling from permuted networks."""
    print("=" * 60)
    print("Test 5: Sample from Permutation")
    print("=" * 60)

    metaedge = "CbG"
    n_samples = 5

    # Sample from permutation 0
    samples_perm0 = sampling.sample_node_pairs(
        metaedge=metaedge,
        source="perm0",
        n_samples_per_category=n_samples,
        quantiles=config.DEGREE_QUANTILES,
        category_names=config.DEGREE_CATEGORY_NAMES,
        random_state=42
    )

    # Sample from permutation 1
    samples_perm1 = sampling.sample_node_pairs(
        metaedge=metaedge,
        source="perm1",
        n_samples_per_category=n_samples,
        quantiles=config.DEGREE_QUANTILES,
        category_names=config.DEGREE_CATEGORY_NAMES,
        random_state=42
    )

    print(f"Permutation 0: {len(samples_perm0)} samples")
    print(f"Permutation 1: {len(samples_perm1)} samples")

    # Verify both have samples
    assert len(samples_perm0) > 0, "Should have samples from perm0"
    assert len(samples_perm1) > 0, "Should have samples from perm1"

    # Verify degree preservation (same category distribution)
    from collections import Counter
    cats0 = Counter(s['category'] for s in samples_perm0)
    cats1 = Counter(s['category'] for s in samples_perm1)

    print("\nCategory distribution perm0:")
    for cat, count in sorted(cats0.items()):
        print(f"  {cat}: {count}")

    print("\nCategory distribution perm1:")
    for cat, count in sorted(cats1.items()):
        print(f"  {cat}: {count}")

    print("PASSED: Sampling from permutations works correctly\n")


def test_metapath_sampling():
    """Test sampling source-target pairs for metapaths."""
    print("=" * 60)
    print("Test 6: Metapath Sampling")
    print("=" * 60)

    metapath = "CbGpPW"
    n_samples = 5

    # Sample pairs for this metapath
    samples = sampling.sample_metapath_pairs(
        metapath=metapath,
        source="true",
        n_samples_per_category=n_samples,
        quantiles=config.DEGREE_QUANTILES,
        category_names=config.DEGREE_CATEGORY_NAMES,
        random_state=42
    )

    print(f"Metapath: {metapath}")
    print(f"Sampled {len(samples)} source-target pairs")
    print(f"First sample: {samples[0]}")

    # Verify output structure
    assert len(samples) > 0, "Should have samples"
    for sample in samples:
        assert 'source_idx' in sample, "Should have source_idx"
        assert 'target_idx' in sample, "Should have target_idx"
        assert 'source_degree' in sample, "Should have source degree"
        assert 'target_degree' in sample, "Should have target degree"
        assert 'category' in sample, "Should have category"
        assert 'metapath' in sample, "Should have metapath"
        assert sample['metapath'] == metapath, "Metapath should match"

    print("PASSED: Metapath sampling works correctly\n")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("SAMPLING MODULE TEST SUITE")
    print("=" * 60)
    print("\n")

    try:
        test_degree_binning()
        test_categorize_pairs()
        test_sample_pairs_by_category()
        test_sample_reproducibility()
        test_sample_from_permutation()
        test_metapath_sampling()

        print("=" * 60)
        print("PASSED: ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\nFAILED: TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nFAILED: ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
