"""
Test script to verify the 6-step label_mapper implementation in EnrichConnectionTablePolars.

This tests the fix for the issue where allpaths_type.csv files were empty when using label_mapper.
"""

import os
import sys
import pandas as pd
import polars as pl

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from statvis_polars import build_bodyid_label_map, get_classification_map, EnrichConnectionTablePolars
from comparison.label_mapper import LabelMapper


def test_build_bodyid_label_map():
    """Test that build_bodyid_label_map correctly expands type mappings to bodyIds."""
    print("\n=== Test: build_bodyid_label_map ===")
    
    # Create a mock label_mapper with type-based mappings
    mapper = LabelMapper(
        source_mapping_dict={
            'hemibrain_v1_2_1': [['5th-LNv', 'LNd_CRY+_ITP+']],
        },
        source_labels=['E cells']
    )
    
    # Create mock neuron DataFrame (like neuron index file)
    neuron_df = pl.DataFrame({
        'bodyId': ['123', '456', '789', '101112'],
        'type': ['5th-LNv', '5th-LNv', 'LNd_CRY+_ITP+', 'OtherType'],
        'instance': ['5th-LNv_R', '5th-LNv_L', 'LNd_CRY+_ITP+', 'OtherInstance']
    })
    
    # Build the bodyId -> label map
    bodyid_map = build_bodyid_label_map(mapper, 'hemibrain_v1_2_1', neuron_df)
    
    print(f"  Input types in mapper: ['5th-LNv', 'LNd_CRY+_ITP+'] -> 'E cells'")
    print(f"  Generated bodyId map: {bodyid_map}")
    
    # Verify that all bodyIds of the mapped types got the label
    assert bodyid_map.get('123') == 'E cells', "BodyId 123 (5th-LNv) should map to 'E cells'"
    assert bodyid_map.get('456') == 'E cells', "BodyId 456 (5th-LNv) should map to 'E cells'"
    assert bodyid_map.get('789') == 'E cells', "BodyId 789 (LNd_CRY+_ITP+) should map to 'E cells'"
    assert '101112' not in bodyid_map or bodyid_map.get('101112') == 'OtherType', "BodyId 101112 should NOT be mapped"
    
    print("  ✓ PASSED: Type-based mappings correctly expanded to bodyIds")
    return True


def test_enrich_connection_table_with_label_mapper():
    """Test that EnrichConnectionTablePolars correctly aggregates using std_labels."""
    print("\n=== Test: EnrichConnectionTablePolars with label_mapper ===")
    
    # Create a mock label_mapper
    mapper = LabelMapper(
        source_mapping_dict={
            'test_dataset': [['TypeA', 'TypeB']],
        },
        source_labels=['Combined_AB']
    )
    
    # Create mock connection table at bodyId level
    conn_table = pl.DataFrame({
        'bodyId_pre': ['1', '2', '3', '4'],
        'bodyId_post': ['5', '6', '7', '8'],
        'type_pre': ['TypeA', 'TypeA', 'TypeB', 'TypeC'],  # TypeA & TypeB should combine
        'type_post': ['TypeD', 'TypeD', 'TypeD', 'TypeE'],
        'weight': [10, 15, 20, 5],
    })
    
    # Create mock neuron dataframe (for total_post calculation)
    neuron_df = pl.DataFrame({
        'bodyId': ['1', '2', '3', '4', '5', '6', '7', '8'],
        'type': ['TypeA', 'TypeA', 'TypeB', 'TypeC', 'TypeD', 'TypeD', 'TypeD', 'TypeE'],
        'post': [100, 100, 100, 100, 200, 200, 200, 150]
    })
    
    # Create a temp script path with the neuron file
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup mock dataset structure
        dataset_dir = os.path.join(temp_dir, 'datasets', 'test_dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        neuron_df.write_csv(os.path.join(dataset_dir, 'test_dataset_allneurons_neuron_df.csv'))
        
        # Run enrichment
        conn_df, conn_type, conn_group = EnrichConnectionTablePolars(
            conn_table,
            dataset='test_dataset',
            script_path=temp_dir,
            label_mapper=mapper
        )
        
        print(f"  Input connection types: TypeA, TypeA, TypeB, TypeC")
        print(f"  Expected: TypeA+TypeB -> Combined_AB, TypeC unchanged")
        print(f"  conn_type result:")
        print(conn_type.to_pandas().to_string())
        
        # Check that TypeA and TypeB are now aggregated under Combined_AB
        type_pre_values = set(conn_type['type_pre'].to_list())
        type_post_values = set(conn_type['type_post'].to_list())
        
        print(f"  Aggregated type_pre values: {type_pre_values}")
        print(f"  Aggregated type_post values: {type_post_values}")
        
        # TypeA and TypeB should now be Combined_AB
        assert 'Combined_AB' in type_pre_values, "Combined_AB should appear in type_pre"
        assert 'TypeA' not in type_pre_values, "TypeA should be replaced by Combined_AB"
        assert 'TypeB' not in type_pre_values, "TypeB should be replaced by Combined_AB"
        assert 'TypeC' in type_pre_values, "TypeC should remain unmapped"
        
        # Check combined weight
        combined_row = conn_type.filter(pl.col('type_pre') == 'Combined_AB')
        if not combined_row.is_empty():
            combined_weight = combined_row['weight'].sum()
            expected_weight = 10 + 15 + 20  # All TypeA and TypeB weights
            assert combined_weight == expected_weight, f"Combined weight should be {expected_weight}, got {combined_weight}"
            print(f"  ✓ Combined weight: {combined_weight} (expected: {expected_weight})")
        
        print("  ✓ PASSED: Label mapper correctly aggregates by std_label")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_without_label_mapper():
    """Test that EnrichConnectionTablePolars works correctly without label_mapper."""
    print("\n=== Test: EnrichConnectionTablePolars without label_mapper ===")
    
    # Create mock connection table
    conn_table = pl.DataFrame({
        'bodyId_pre': ['1', '2', '3'],
        'bodyId_post': ['4', '5', '6'],
        'type_pre': ['TypeA', 'TypeA', 'TypeB'],
        'type_post': ['TypeC', 'TypeC', 'TypeD'],
        'weight': [10, 15, 20],
    })
    
    # Run enrichment without label_mapper
    conn_df, conn_type, conn_group = EnrichConnectionTablePolars(conn_table)
    
    print(f"  Input types: TypeA, TypeA, TypeB")
    print(f"  conn_type result:")
    print(conn_type.to_pandas().to_string())
    
    type_pre_values = set(conn_type['type_pre'].to_list())
    
    # Should use original types
    assert 'TypeA' in type_pre_values, "TypeA should be preserved"
    assert 'TypeB' in type_pre_values, "TypeB should be preserved"
    
    print("  ✓ PASSED: Without label_mapper, original types are preserved")
    return True


def test_bodyid_fallback_for_untyped_neurons():
    """Test that untyped neurons use bodyId as fallback label."""
    print("\n=== Test: BodyId fallback for untyped neurons ===")
    
    # Create connection table with some null/empty types
    conn_table = pl.DataFrame({
        'bodyId_pre': ['1', '2', '3', '4'],
        'bodyId_post': ['5', '6', '7', '8'],
        'type_pre': ['TypeA', '', None, 'TypeB'],  # bodyIds 2 and 3 have no type
        'type_post': ['TypeC', 'TypeC', '', 'TypeD'],  # bodyId 7 has no type
        'weight': [10, 15, 20, 5],
    })
    
    # Run enrichment without label_mapper
    conn_df, conn_type, conn_group = EnrichConnectionTablePolars(conn_table)
    
    print(f"  Input: bodyId 2 has empty type, bodyId 3 has null type, bodyId 7 has empty type")
    print(f"  conn_type result:")
    print(conn_type.to_pandas().to_string())
    
    type_pre_values = set(conn_type['type_pre'].to_list())
    type_post_values = set(conn_type['type_post'].to_list())
    
    print(f"  type_pre values: {type_pre_values}")
    print(f"  type_post values: {type_post_values}")
    
    # Should use bodyId for untyped neurons
    assert '2' in type_pre_values or '3' in type_pre_values, "BodyId should be used for untyped neurons"
    assert '' not in type_pre_values, "Empty string should NOT appear in type_pre"
    assert '7' in type_post_values, "BodyId 7 should be used for untyped neuron"
    assert '' not in type_post_values, "Empty string should NOT appear in type_post"
    
    print("  ✓ PASSED: BodyId is used as fallback for untyped neurons")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing 6-Step Label Mapper Implementation")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_build_bodyid_label_map()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_without_label_mapper()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_enrich_connection_table_with_label_mapper()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_bodyid_fallback_for_untyped_neurons()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)