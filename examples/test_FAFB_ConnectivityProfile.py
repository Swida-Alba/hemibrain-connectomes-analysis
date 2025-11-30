"""
Test script for FAFB/FlyWire Connectivity Profile Extraction

This test verifies that the ConnectivityProfiler correctly extracts
connectivity profiles from FlyWire FAFB datasets.

The key issue being tested: FAFB neuron_df has two type columns:
- 'type': Specific neuron types like 'aMe12', 'PPL101', 'T5c'
- 'cell_type': Generic categories like 't5_neuron', 'transmedullary'

The profiler should use 'type' (specific) not 'cell_type' (generic).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comparison import ConnectivityProfiler, ProfilerConfig


def test_fafb_connectivity_profile():
    """Test that FAFB connectivity profiles are extracted correctly."""
    print("=" * 60)
    print("Test: FAFB Connectivity Profile Extraction")
    print("=" * 60)
    
    # Create profiler with FAFB dataset
    config = ProfilerConfig(
        top_k_upstream=15,
        top_k_downstream=15,
        min_synapse_threshold=3,
        include_untyped_partners=False,
        use_cache=False  # Don't use cache for testing
    )
    
    profiler = ConnectivityProfiler(
        datasets=['flywire_FAFB_v783'],
        config=config,
        verbose=True
    )
    
    # Test 1: Extract profile for aMe12
    print("\n" + "-" * 40)
    print("Test 1: Extract profile for aMe12")
    print("-" * 40)
    
    profile_ame12 = profiler.get_profile('aMe12', 'flywire_FAFB_v783')
    
    print(f"\naMe12 Profile Summary:")
    print(f"  Dataset: {profile_ame12.dataset}")
    print(f"  Neuron ID: {profile_ame12.neuron_id}")
    print(f"  Total upstream weight: {profile_ame12.total_upstream_weight}")
    print(f"  Total downstream weight: {profile_ame12.total_downstream_weight}")
    print(f"  Upstream partner count: {len(profile_ame12.upstream_partners)}")
    print(f"  Downstream partner count: {len(profile_ame12.downstream_partners)}")
    print(f"  Is weak connectivity: {profile_ame12.is_weak_connectivity}")
    
    # Check that we got non-zero connections
    assert len(profile_ame12.upstream_partners) > 0, "ERROR: aMe12 should have upstream partners"
    assert len(profile_ame12.downstream_partners) > 0, "ERROR: aMe12 should have downstream partners"
    print("\n✓ aMe12 has non-zero upstream and downstream partners")
    
    # Print top partners
    print("\nTop upstream partners:")
    for partner, weight in sorted(profile_ame12.upstream_partners.items(), key=lambda x: -x[1])[:10]:
        print(f"  {partner}: {weight:.3f}")
    
    print("\nTop downstream partners:")
    for partner, weight in sorted(profile_ame12.downstream_partners.items(), key=lambda x: -x[1])[:10]:
        print(f"  {partner}: {weight:.3f}")
    
    # Test 2: Extract profile for PPL101
    print("\n" + "-" * 40)
    print("Test 2: Extract profile for PPL101")
    print("-" * 40)
    
    profile_ppl101 = profiler.get_profile('PPL101', 'flywire_FAFB_v783')
    
    print(f"\nPPL101 Profile Summary:")
    print(f"  Upstream partner count: {len(profile_ppl101.upstream_partners)}")
    print(f"  Downstream partner count: {len(profile_ppl101.downstream_partners)}")
    print(f"  Is weak connectivity: {profile_ppl101.is_weak_connectivity}")
    
    # PPL101 may have weak connectivity due to only 2 neurons, but should not be all zeros
    assert profile_ppl101.total_upstream_weight > 0 or profile_ppl101.total_downstream_weight > 0, \
        "ERROR: PPL101 should have some connections"
    print("\n✓ PPL101 has connections")
    
    # Test 3: Extract profile for T5c (common optic type)
    print("\n" + "-" * 40)
    print("Test 3: Extract profile for T5c")
    print("-" * 40)
    
    profile_t5c = profiler.get_profile('T5c', 'flywire_FAFB_v783')
    
    print(f"\nT5c Profile Summary:")
    print(f"  Upstream partner count: {len(profile_t5c.upstream_partners)}")
    print(f"  Downstream partner count: {len(profile_t5c.downstream_partners)}")
    print(f"  Is weak connectivity: {profile_t5c.is_weak_connectivity}")
    
    assert len(profile_t5c.upstream_partners) > 0, "ERROR: T5c should have upstream partners"
    print("\n✓ T5c has non-zero upstream partners")
    
    # Test 4: Verify type column is being used correctly (not cell_type)
    print("\n" + "-" * 40)
    print("Test 4: Verify correct type column usage")
    print("-" * 40)
    
    # Check that partner types are specific (like 'R8', 'Dm9') not generic (like 'photo_receptor')
    all_partners = set(profile_ame12.upstream_partners.keys()) | set(profile_ame12.downstream_partners.keys())
    
    # These are generic cell_type categories that should NOT appear if we're using 'type' correctly
    generic_types = {'t5_neuron', 't4_neuron', 'transmedullary', 'photo_receptor', 'distal_medulla'}
    
    # These are specific types that SHOULD appear
    specific_types = {'R8', 'R7', 'Dm9', 'KCg-d', 'aMe24'}
    
    found_generic = all_partners & generic_types
    found_specific = all_partners & specific_types
    
    print(f"Partner types found: {len(all_partners)}")
    print(f"Generic cell_type found (should be empty): {found_generic}")
    print(f"Specific types found: {found_specific}")
    
    if found_generic:
        print("\n⚠️  WARNING: Found generic cell_type categories - may be using wrong column!")
    else:
        print("\n✓ No generic cell_type categories found - using correct 'type' column")
    
    assert len(found_generic) == 0, f"ERROR: Found generic cell_type categories: {found_generic}"
    
    print("\n" + "=" * 60)
    print("All tests passed! FAFB connectivity profiles work correctly.")
    print("=" * 60)
    
    return True


def test_fafb_cross_dataset_comparison():
    """Test that FAFB profiles can be compared with other datasets."""
    print("\n" + "=" * 60)
    print("Test: Cross-Dataset Comparison with FAFB")
    print("=" * 60)
    
    from src.comparison import CrossDatasetVerifier
    
    config = ProfilerConfig(
        top_k_upstream=15,
        top_k_downstream=15,
        min_synapse_threshold=3,
        use_cache=False
    )
    
    profiler = ConnectivityProfiler(
        datasets=['hemibrain:v1.2.1', 'flywire_FAFB_v783'],
        config=config,
        verbose=True
    )
    
    verifier = CrossDatasetVerifier(profiler, verbose=True)
    
    # Test verification for aMe12
    print("\n" + "-" * 40)
    print("Verifying aMe12 across hemibrain and FAFB")
    print("-" * 40)
    
    result = verifier.verify_type_assignment(
        'aMe12',
        datasets=['hemibrain:v1.2.1', 'flywire_FAFB_v783']
    )
    
    print(f"\nVerification Result:")
    print(result.summary())
    
    # The score should be non-zero now
    assert result.avg_combined_score > 0, \
        f"ERROR: aMe12 combined score should be > 0, got {result.avg_combined_score}"
    print(f"\n✓ aMe12 cross-dataset score: {result.avg_combined_score:.3f}")
    
    # Test verification for multiple types
    print("\n" + "-" * 40)
    print("Batch verification for multiple types")
    print("-" * 40)
    
    types_to_test = ['aMe12', 'PPL101', 'T5c', 'Dm9']
    
    results_df = verifier.batch_verify_types(
        types_to_test,
        datasets=['hemibrain:v1.2.1', 'flywire_FAFB_v783']
    )
    
    print("\nBatch verification results:")
    print(results_df.to_string(index=False))
    
    # Check that at least some types have non-zero scores
    non_zero_count = (results_df['avg_combined_score'] > 0).sum()
    print(f"\n✓ {non_zero_count}/{len(types_to_test)} types have non-zero cross-dataset scores")
    
    assert non_zero_count > 0, "ERROR: At least some types should have non-zero scores"
    
    print("\n" + "=" * 60)
    print("Cross-dataset comparison tests passed!")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    print("\n" + "#" * 70)
    print("# FAFB/FlyWire Connectivity Profile Test Suite")
    print("#" * 70)
    
    # Run tests
    success = True
    
    try:
        test_fafb_connectivity_profile()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        test_fafb_cross_dataset_comparison()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "#" * 70)
    if success:
        print("# ✅ ALL TESTS PASSED")
    else:
        print("# ❌ SOME TESTS FAILED")
    print("#" * 70)
