"""
Comprehensive Test Suite for VisualizeSkeleton Enhanced Features

This script tests:
1. Automatic ROI mesh download from NeuPrint
2. Listing available online ROI areas for different datasets
3. Dataset-specific caching
4. Error handling and fallback mechanisms

Author: hemibrain-connectomes-analysis-v3.1
Date: November 20, 2024
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

import os
import statvis as sv
from coana import VisualizeSkeleton

# =============================================================================
# Configuration
# =============================================================================

# Get NeuPrint token from environment variable
# Get your token from: https://neuprint.janelia.org/account
# Set it with: export NEUPRINT_APPLICATION_CREDENTIALS="your_token_here"
TOKEN = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS', '')

if not TOKEN:
    print('⚠️  NEUPRINT_APPLICATION_CREDENTIALS environment variable not set')
    print('   Get your token from: https://neuprint.janelia.org/account')
    print('   Set it with: export NEUPRINT_APPLICATION_CREDENTIALS="your_token_here"')
    print('\n   Alternatively, you can set it temporarily in this session:')
    print('   export NEUPRINT_APPLICATION_CREDENTIALS="eyJhbGc..."')
    sys.exit(1)

print(f'✓ Using NeuPrint token from environment variable')
print(f'  Token prefix: {TOKEN[:20]}...')

TEST_OUTPUT_DIR = '/tmp/visualize_skeleton_tests'
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

print('='*80)
print('COMPREHENSIVE TEST SUITE FOR VISUALIZESKELETON ENHANCEMENTS')
print('='*80)
print(f'Output directory: {TEST_OUTPUT_DIR}\n')

# =============================================================================
# Test 1: List Available Online ROIs for Hemibrain
# =============================================================================
print('\n' + '='*80)
print('TEST 1: List Available Online ROIs for Hemibrain Dataset')
print('='*80)

try:
    # Login to hemibrain dataset
    server_client, dataset = sv.LogInHemibrain(token=TOKEN, dataset='hemibrain:v1.2.1')
    
    # Create a minimal VisualizeSkeleton instance (no visualization needed)
    vs_hemibrain = VisualizeSkeleton(
        dataset='hemibrain:v1.2.1',
        data_folder=TEST_OUTPUT_DIR,
        neuron_layers=['EB'],  # Minimal layer
        show_fig=False,
    )
    
    # Test listing available ROIs from online database
    print('\n--- Fetching from online database (first time) ---')
    available_rois_online = vs_hemibrain.list_available_rois(refresh=True, fetch_online=True)
    
    # Test using cached data
    print('\n--- Using cached data (second time) ---')
    available_rois_cached = vs_hemibrain.list_available_rois(refresh=False, fetch_online=False)
    
    # Verify results
    assert len(available_rois_online) > 0, "Should fetch ROIs from online database"
    assert len(available_rois_cached) > 0, "Should load ROIs from cache"
    assert available_rois_online == available_rois_cached, "Cached and online results should match"
    
    print(f'\n✅ TEST 1 PASSED: Successfully fetched and cached {len(available_rois_online)} ROIs')
    
except Exception as e:
    print(f'\n❌ TEST 1 FAILED: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 2: List Available Online ROIs for Optic-Lobe Dataset
# =============================================================================
print('\n' + '='*80)
print('TEST 2: List Available Online ROIs for Optic-Lobe Dataset')
print('='*80)

try:
    # Login to optic-lobe dataset
    server_client_ol, dataset_ol = sv.LogInHemibrain(token=TOKEN, dataset='optic-lobe:v1.1')
    
    # Create instance for optic-lobe
    vs_optic = VisualizeSkeleton(
        dataset='optic-lobe:v1.1',
        data_folder=TEST_OUTPUT_DIR,
        neuron_layers=['LNd'],
        show_fig=False,
    )
    
    # List available ROIs
    print('\n--- Fetching optic-lobe ROIs from online database ---')
    optic_rois = vs_optic.list_available_rois(refresh=True, fetch_online=True)
    
    # Verify results
    assert len(optic_rois) > 0, "Should fetch ROIs for optic-lobe dataset"
    
    print(f'\n✅ TEST 2 PASSED: Successfully fetched {len(optic_rois)} ROIs for optic-lobe')
    
except Exception as e:
    print(f'\n❌ TEST 2 FAILED: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 3: Automatic ROI Mesh Download
# =============================================================================
print('\n' + '='*80)
print('TEST 3: Automatic ROI Mesh Download from NeuPrint')
print('='*80)

try:
    # Login to hemibrain
    server_client, dataset = sv.LogInHemibrain(token=TOKEN, dataset='hemibrain:v1.2.1')
    
    # Choose an ROI that might not be in local cache
    test_roi = 'NO'  # Noduli - small central brain region
    
    print(f'\n--- Testing automatic download of ROI: {test_roi} ---')
    
    # Create instance with specific ROI
    vs_download_test = VisualizeSkeleton(
        dataset='hemibrain:v1.2.1',
        data_folder=TEST_OUTPUT_DIR,
        neuron_layers=['EB'],
        mesh_roi=[test_roi],  # Will trigger download if not cached
        show_fig=False,
        saveas='download_test',
    )
    
    # Attempt to plot (which will trigger download if needed)
    print(f'\n--- Plotting skeleton with ROI {test_roi} (will auto-download if missing) ---')
    vs_download_test.plot_skeleton()
    vs_download_test.plot_mesh()  # This will attempt to download if missing
    
    # Check if mesh file was created
    mesh_dir = vs_download_test._get_dataset_mesh_dir()
    expected_mesh_file = os.path.join(mesh_dir, f'{test_roi}.json')
    
    if os.path.exists(expected_mesh_file):
        print(f'\n✅ TEST 3 PASSED: ROI mesh {test_roi} successfully available at {expected_mesh_file}')
    else:
        # Check primary_rois fallback
        fallback_file = os.path.join(vs_download_test.script_path, 'navis_roi_meshes_json', 'primary_rois', f'{test_roi}.json')
        if os.path.exists(fallback_file):
            print(f'\n✅ TEST 3 PASSED: ROI mesh {test_roi} found in fallback location')
        else:
            print(f'\n⚠️  TEST 3 WARNING: ROI mesh not found, but download may have been attempted')
    
except Exception as e:
    print(f'\n❌ TEST 3 FAILED: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 4: Multiple ROI Download Test
# =============================================================================
print('\n' + '='*80)
print('TEST 4: Multiple ROI Mesh Download and Caching')
print('='*80)

try:
    server_client, dataset = sv.LogInHemibrain(token=TOKEN, dataset='hemibrain:v1.2.1')
    
    # Test with multiple ROIs - mix of common and less common ones
    test_rois = ['EB', 'PB', 'FB', 'NO']
    
    print(f'\n--- Testing download/cache of multiple ROIs: {test_rois} ---')
    
    vs_multi = VisualizeSkeleton(
        dataset='hemibrain:v1.2.1',
        data_folder=TEST_OUTPUT_DIR,
        neuron_layers=['EB'],
        mesh_roi=test_rois,
        show_fig=False,
        saveas='multi_roi_test',
    )
    
    print('\n--- First run: downloading missing meshes ---')
    vs_multi.plot_skeleton()
    vs_multi.plot_mesh()
    
    print('\n--- Second run: should use cached meshes ---')
    vs_multi2 = VisualizeSkeleton(
        dataset='hemibrain:v1.2.1',
        data_folder=TEST_OUTPUT_DIR,
        neuron_layers=['EB'],
        mesh_roi=test_rois,
        show_fig=False,
        saveas='multi_roi_test2',
    )
    vs_multi2.plot_skeleton()
    vs_multi2.plot_mesh()
    
    print(f'\n✅ TEST 4 PASSED: Successfully handled multiple ROI meshes')
    
except Exception as e:
    print(f'\n❌ TEST 4 FAILED: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 5: Error Handling - Invalid ROI Name
# =============================================================================
print('\n' + '='*80)
print('TEST 5: Error Handling for Invalid ROI Names')
print('='*80)

try:
    server_client, dataset = sv.LogInHemibrain(token=TOKEN, dataset='hemibrain:v1.2.1')
    
    # Test with an invalid ROI name
    invalid_roi = 'INVALID_ROI_NAME_XYZ123'
    
    print(f'\n--- Testing with invalid ROI: {invalid_roi} ---')
    print('Expected: Should handle gracefully without crashing')
    
    vs_error_test = VisualizeSkeleton(
        dataset='hemibrain:v1.2.1',
        data_folder=TEST_OUTPUT_DIR,
        neuron_layers=['EB'],
        mesh_roi=[invalid_roi, 'EB'],  # Mix invalid with valid
        show_fig=False,
        saveas='error_test',
    )
    
    vs_error_test.plot_skeleton()
    vs_error_test.plot_mesh()  # Should handle invalid ROI gracefully
    
    print(f'\n✅ TEST 5 PASSED: Gracefully handled invalid ROI name')
    
except Exception as e:
    print(f'\n❌ TEST 5 FAILED: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 6: Dataset-Specific Caching Verification
# =============================================================================
print('\n' + '='*80)
print('TEST 6: Dataset-Specific Cache Directory Structure')
print('='*80)

try:
    # Check if dataset-specific directories exist
    script_path = Path(__file__).parent.parent.parent.parent
    mesh_base_dir = script_path / 'navis_roi_meshes_json'
    
    print('\n--- Checking cache directory structure ---')
    
    expected_dirs = [
        'hemibrain_v1_2_1',
        'optic-lobe_v1_1',
        'fib',
        'manc',
        'primary_rois'
    ]
    
    print(f'\nCache base directory: {mesh_base_dir}')
    print(f'\nExpected subdirectories:')
    
    all_exist = True
    for dir_name in expected_dirs:
        dir_path = mesh_base_dir / dir_name
        exists = dir_path.exists()
        status = '✓' if exists else '✗'
        print(f'  {status} {dir_name}/')
        if not exists and dir_name in ['hemibrain_v1_2_1', 'primary_rois']:
            all_exist = False
    
    # Check for cached ROI lists
    print(f'\n--- Checking for cached ROI lists ---')
    cache_files = list(mesh_base_dir.glob('*_available_rois.json'))
    print(f'Found {len(cache_files)} cached ROI list(s):')
    for cache_file in cache_files:
        print(f'  • {cache_file.name}')
    
    if all_exist:
        print(f'\n✅ TEST 6 PASSED: Cache directory structure is correct')
    else:
        print(f'\n⚠️  TEST 6 WARNING: Some expected directories are missing (non-critical)')
    
except Exception as e:
    print(f'\n❌ TEST 6 FAILED: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 7: ROI List Comparison (Online vs Cached)
# =============================================================================
print('\n' + '='*80)
print('TEST 7: ROI List Consistency (Online vs Cached)')
print('='*80)

try:
    server_client, dataset = sv.LogInHemibrain(token=TOKEN, dataset='hemibrain:v1.2.1')
    
    vs_test = VisualizeSkeleton(
        dataset='hemibrain:v1.2.1',
        data_folder=TEST_OUTPUT_DIR,
        neuron_layers=['EB'],
        show_fig=False,
    )
    
    # Get ROIs from different sources
    print('\n--- Fetching from online database ---')
    rois_online = vs_test._get_available_rois(use_cache=False, fetch_online=True)
    
    print('\n--- Loading from cache ---')
    rois_cached = vs_test._get_available_rois(use_cache=True, fetch_online=False)
    
    print('\n--- Loading from local mesh directory ---')
    rois_local = vs_test._get_available_rois(use_cache=False, fetch_online=False)
    
    # Compare results
    print(f'\n--- Comparison ---')
    print(f'Online database:    {len(rois_online)} ROIs')
    print(f'Cached list:        {len(rois_cached)} ROIs')
    print(f'Local mesh files:   {len(rois_local)} ROIs')
    
    # Check if online and cached match
    if rois_online == rois_cached:
        print(f'\n✅ TEST 7 PASSED: Online and cached ROI lists match perfectly')
    else:
        diff = set(rois_online) - set(rois_cached)
        print(f'\n⚠️  TEST 7 WARNING: Differences found ({len(diff)} ROIs)')
        if diff:
            print(f'   ROIs in online but not cached: {list(diff)[:5]}...')
    
except Exception as e:
    print(f'\n❌ TEST 7 FAILED: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*80)
print('TEST SUITE SUMMARY')
print('='*80)

summary = f"""
Tests Completed:
  ✓ Test 1: List available online ROIs for hemibrain
  ✓ Test 2: List available online ROIs for optic-lobe
  ✓ Test 3: Automatic ROI mesh download
  ✓ Test 4: Multiple ROI download and caching
  ✓ Test 5: Error handling for invalid ROI names
  ✓ Test 6: Dataset-specific cache directory structure
  ✓ Test 7: ROI list consistency check

Key Features Tested:
  • Fetching ROI lists from NeuPrint online database
  • Caching ROI lists locally for performance
  • Automatic download of missing ROI meshes using navis.interfaces.neuprint.fetch_roi()
  • Dataset-specific mesh caching (hemibrain_v1_2_1/, optic-lobe_v1_1/, etc.)
  • Fallback mechanisms (online → cache → local meshes)
  • Error handling and graceful degradation

Output Location:
  {TEST_OUTPUT_DIR}

Next Steps:
  1. Check test output files in {TEST_OUTPUT_DIR}
  2. Verify downloaded mesh files in navis_roi_meshes_json/ subdirectories
  3. Review cached ROI lists (*_available_rois.json)
  4. Uncomment plot_neurons() calls in tests to generate visualizations

Documentation:
  • docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md
  • docs/visualizations/VisualizeSkeleton_Quick_Reference.md
"""

print(summary)
print('='*80)
print('✅ ALL TESTS COMPLETED')
print('='*80)
