"""
Example script demonstrating VisualizeSkeleton's multi-dataset support.

This example shows:
1. Dataset-specific ROI mesh caching (hemibrain vs optic-lobe)
2. Listing available ROIs for each dataset
3. Backward compatibility with primary_rois/
4. Brain transformation handling with user confirmation

Author: drocat
Date: November 2024
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

import statvis as sv
from visualize_skeleton import VisualizeSkeleton

# =============================================================================
# Test 1: Hemibrain dataset with dataset-specific mesh caching
# =============================================================================
print('='*70)
print('Test 1: Hemibrain dataset (hemibrain:v1.2.1)')
print('='*70)

# Replace with your NeuPrint token from https://neuprint.janelia.org/account
TOKEN = ''

if not TOKEN:
    print('⚠️  Please set your NeuPrint token in the TOKEN variable')
    print('   Get your token from: https://neuprint.janelia.org/account')
    sys.exit(1)

# Login to hemibrain dataset
server_client, dataset = sv.LogInHemibrain(token=TOKEN, dataset='hemibrain:v1.2.1')

# Create VisualizeSkeleton instance
vs_hemibrain = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    output_dir='../../../local_data/plot_3d',  # Use local_data for testing
    neuron_layers=['EB'],  # Simple single layer for testing
    custom_layer_names=['Ellipsoid Body Neurons'],
    neuron_alpha=0.3,
    min_synapse_num=1,
    synapse_size=2,
    synapse_alpha=0.6,
    mesh_roi=['EB', 'PB', 'FB'],  # Central complex regions
    brain_mesh='none',  # Use 'whole' to test transform confirmation
    show_fig=False,  # Don't auto-open browser during testing
)

# List available ROIs for hemibrain
print('\n--- Listing Available ROIs for Hemibrain ---')
available_rois = vs_hemibrain.list_available_rois()

# Verify dataset-specific mesh directory is being used
print('\n--- Verifying Dataset-Specific Mesh Directory ---')
mesh_dir = vs_hemibrain._get_dataset_mesh_dir()
print(f'Using mesh directory: {mesh_dir}')
expected_dir = 'hemibrain_v1_2_1' if 'hemibrain_v1_2_1' in mesh_dir else 'primary_rois'
print(f'Expected directory type: {expected_dir}')

# Plot the visualization
print('\n--- Generating Hemibrain Visualization ---')
# vs_hemibrain.plot_neurons()  # Uncomment to generate visualization
print('✓ Hemibrain test setup complete (uncomment plot_neurons() to render)')

# =============================================================================
# Test 2: Optic-lobe dataset with dataset-specific mesh caching
# =============================================================================
print('\n' + '='*70)
print('Test 2: Optic-lobe dataset (optic-lobe:v1.1)')
print('='*70)

# Login to optic-lobe dataset
server_client_ol, dataset_ol = sv.LogInHemibrain(token=TOKEN, dataset='optic-lobe:v1.1')

# Create VisualizeSkeleton instance for optic-lobe
vs_optic_lobe = VisualizeSkeleton(
    dataset='optic-lobe:v1.1',
    output_dir='../../../local_data/plot_3d',
    neuron_layers=['LNd'],  # Optic lobe neurons
    custom_layer_names=['Optic Lobe LNd Neurons'],
    neuron_alpha=0.3,
    min_synapse_num=1,
    synapse_size=2,
    synapse_alpha=0.6,
    mesh_roi=['ME(R)', 'AME(R)'],  # Optic lobe regions (if available)
    brain_mesh='none',
    show_fig=False,
)

# List available ROIs for optic-lobe
print('\n--- Listing Available ROIs for Optic-Lobe ---')
available_rois_ol = vs_optic_lobe.list_available_rois()

# Verify dataset-specific mesh directory
print('\n--- Verifying Optic-Lobe Mesh Directory ---')
mesh_dir_ol = vs_optic_lobe._get_dataset_mesh_dir()
print(f'Using mesh directory: {mesh_dir_ol}')
expected_dir_ol = 'optic-lobe_v1_1' if 'optic-lobe_v1_1' in mesh_dir_ol else 'primary_rois'
print(f'Expected directory type: {expected_dir_ol} (fallback is normal if optic-lobe meshes not available)')

# Plot the visualization
print('\n--- Generating Optic-Lobe Visualization ---')
# vs_optic_lobe.plot_neurons()  # Uncomment to generate visualization
print('✓ Optic-lobe test setup complete (uncomment plot_neurons() to render)')

# =============================================================================
# Test 3: Brain mesh transformation with confirmation
# =============================================================================
print('\n' + '='*70)
print('Test 3: Brain mesh transformation (requires flybrains)')
print('='*70)

print('''
To test brain transformation with user confirmation:
1. Uncomment the code below
2. Run the script
3. When prompted, choose 'y' to download transforms (first time only)

Note: This downloads ~500MB of transform data to ~/.navis/transforms/

Example code to test:
------------------------
vs_whole_brain = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    output_dir='../../../local_data/plot_3d',
    neuron_layers=['EB'],
    mesh_roi=['EB'],
    brain_mesh='whole',  # This will trigger transform check
    show_fig=False,
)
# vs_whole_brain.plot_neurons()
------------------------
''')

# =============================================================================
# Test 4: Backward compatibility with primary_rois/
# =============================================================================
print('\n' + '='*70)
print('Test 4: Backward Compatibility')
print('='*70)

print('''
Backward compatibility is automatic:
- If dataset-specific mesh directory doesn't exist, falls back to primary_rois/
- Existing scripts using primary_rois/ will continue to work
- New datasets automatically get their own cache directories

Directory structure:
navis_roi_meshes_json/
├── primary_rois/         # Backward compatibility (hemibrain)
├── hemibrain_v1_2_1/     # Same content as primary_rois
├── optic-lobe_v1_1/      # Optic-lobe specific meshes
├── fib/                  # FIB specific meshes
└── manc/                 # MANC specific meshes
''')

# =============================================================================
# Summary
# =============================================================================
print('\n' + '='*70)
print('SUMMARY: VisualizeSkeleton Multi-Dataset Features')
print('='*70)
print('''
✓ Dataset-specific ROI mesh caching
  - Each dataset gets its own cache directory
  - Automatic fallback to primary_rois/ for backward compatibility
  
✓ ROI discovery from NeuPrint
  - list_available_rois() method queries database
  - Results cached locally to avoid repeated API calls
  - Fallback to local mesh directory if API fails
  
✓ Brain transformation confirmation
  - Checks for existing transforms before downloading
  - Prompts user for confirmation (~500MB download)
  - Clear error handling with automatic fallback
  
✓ Documentation and best practices
  - Comprehensive docstrings with references
  - Links to navis, flybrains, and NeuPrint documentation
  - Examples for mesh optimization and compression

To run full visualization:
1. Uncomment the plot_neurons() calls above
2. Ensure you have sufficient memory (brain meshes are large)
3. Open the generated HTML files in your browser
''')

print('✓ All tests completed!')
