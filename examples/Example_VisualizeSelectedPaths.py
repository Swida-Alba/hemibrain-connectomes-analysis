"""
Example: Visualizing Selected Neural Pathways (Standalone Usage)

This script demonstrates how to use the VisualizePath class to create
focused visualizations of specific neural pathways discovered through
connectome analysis.

The VisualizePath class can be used in two ways:
1. Standalone - Without initializing FindNeuronConnection (RECOMMENDED)
2. Through FindNeuronConnection - Using the convenience wrapper method

Author: Kun-Da Wu
Date: 2025-10-27
"""

import sys
from pathlib import Path

# Add vispath-subproject to Python path for local development
vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

import pandas as pd

from vispath_pkg import VisualizePath, visualize_paths

# =============================================================================
# Method 1: Standalone Usage - Direct VisualizePath Class
# =============================================================================
print("="*80)
print("Method 1: Standalone Usage - No FindNeuronConnection needed")
print("="*80)

# This is the recommended way when you only want visualization
# You don't need to initialize FindNeuronConnection first!

vp = VisualizePath(
    path_file='./path_results/path_type.xlsx',
    sheet_name='path_type',  # or 'path_bodyId'
    output_folder='./standalone_visualization',
    network_layout='hierarchical',
    showfig=False
)

conn_df, G = vp.visualize()

print(f"\n✓ Created {len(conn_df)} connections")
print(f"✓ Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# =============================================================================
# Method 2: Convenience Function - Quick One-Liner
# =============================================================================
print("\n" + "="*80)
print("Method 2: Convenience Function - Quick One-Liner")
print("="*80)

# Even simpler - just call the function!
conn_df, G = visualize_paths(
    path_file='./path_results/path_type.xlsx',
    output_folder='./quick_viz',
    showfig=True  # Opens in browser automatically
)

# =============================================================================
# Method 3: Filter High-Quality Paths Before Visualization
# =============================================================================
print("\n" + "="*80)
print("Method 3: Filter Then Visualize - Best Practice")
print("="*80)

# Read all paths from FindAllPath results
all_paths = pd.read_excel('./path_results/path_type.xlsx', sheet_name='path_type')

print(f"Total paths found: {len(all_paths)}")

# Filter: Keep only high-probability paths with short hops
high_quality = all_paths[
    (all_paths['traversal_probability'] > 0.5) &
    (all_paths['inter_layer_num'] <= 2)
]

print(f"High-quality paths: {len(high_quality)}")

# Visualize the filtered paths
vp = VisualizePath(
    path_file=high_quality,  # Can pass DataFrame directly!
    output_folder='./high_quality_paths',
    network_layout='hierarchical',
    showfig=False
)

conn_df, G = vp.visualize()

# =============================================================================
# Method 4: Custom Colors and Layout
# =============================================================================
print("\n" + "="*80)
print("Method 4: Custom Colors and Layout")
print("="*80)

vp = VisualizePath(
    path_file='./path_results/path_type.xlsx',
    output_folder='./custom_style',
    node_color=['#FF6B6B', '#FFA500'],  # Red-orange theme
    target_color='#FFD700',              # Gold
    link_color='rgba(255,107,107,0.3)', # Semi-transparent red
    network_layout='spring',             # Force-directed layout
    showfig=True
)

conn_df, G = vp.visualize()

# =============================================================================
# Method 5: Programmatic Path Selection
# =============================================================================
print("\n" + "="*80)
print("Method 5: Programmatic Path Selection")
print("="*80)

# Read all paths
all_paths = pd.read_excel('./path_results/path_type.xlsx', sheet_name='path_type')

# Select paths through specific intermediate neuron types
intermediate_type = 'Mi1'
paths_through_mi1 = all_paths[
    all_paths['path_block'].str.contains(intermediate_type)
]

print(f"Paths through {intermediate_type}: {len(paths_through_mi1)}")

# Visualize
vp = VisualizePath(
    path_file=paths_through_mi1,
    output_folder=f'./paths_via_{intermediate_type}',
    showfig=False
)

conn_df, G = vp.visualize()

# =============================================================================
# Method 6: Create Manual Path Dataset
# =============================================================================
print("\n" + "="*80)
print("Method 6: Create Manual Path Dataset")
print("="*80)

# Manually create a path dataset (useful for testing or specific cases)
manual_paths = pd.DataFrame({
    'path_block': [
        'L3_R -> Mi1_R -> Tm3_R -> T4a_R',
        'L3_R -> Mi4_R -> Tm3_R -> T4a_R',
        'L3_R -> Mi1_R -> TmY3_R -> T4a_R'
    ],
    'weights': [
        [150, 80, 45],
        [120, 65, 40],
        [150, 70, 35]
    ],
    'connection_ratios': [
        [0.25, 0.18, 0.12],
        [0.22, 0.16, 0.11],
        [0.25, 0.15, 0.10]
    ],
    'traversal_probabilities': [
        [0.85, 0.75, 0.65],
        [0.82, 0.70, 0.60],
        [0.85, 0.68, 0.55]
    ]
})

# Visualize
vp = VisualizePath(
    path_file=manual_paths,
    output_folder='./manual_paths',
    showfig=False
)

conn_df, G = vp.visualize()

# =============================================================================
# Method 7: Compare Different Path Sets
# =============================================================================
print("\n" + "="*80)
print("Method 7: Compare Different Path Sets")
print("="*80)

# Scenario: Compare strong vs weak connections

all_paths = pd.read_excel('./path_results/path_type.xlsx', sheet_name='path_type')

# Strong connections (high weight)
strong_paths = all_paths[all_paths['min_weight'] > 100]
print(f"Strong connections: {len(strong_paths)}")

vp_strong = VisualizePath(
    path_file=strong_paths,
    output_folder='./comparison/strong_connections',
    node_color=['#2E7D32', '#66BB6A'],  # Green theme
    target_color='#1B5E20',
    showfig=False
)
vp_strong.visualize()

# Weak connections (low weight but present)
weak_paths = all_paths[
    (all_paths['min_weight'] >= 10) & 
    (all_paths['min_weight'] <= 30)
]
print(f"Weak connections: {len(weak_paths)}")

vp_weak = VisualizePath(
    path_file=weak_paths,
    output_folder='./comparison/weak_connections',
    node_color=['#1565C0', '#42A5F5'],  # Blue theme
    target_color='#0D47A1',
    showfig=False
)
vp_weak.visualize()

# =============================================================================
# Method 8: Advanced Filtering with Multiple Criteria
# =============================================================================
print("\n" + "="*80)
print("Method 8: Advanced Filtering")
print("="*80)

all_paths = pd.read_excel('./path_results/path_type.xlsx', sheet_name='path_type')

# Complex filtering:
# 1. High minimum weight (strong connections)
# 2. High traversal probability
# 3. Specific path length
# 4. Through specific neuron types

advanced_filter = all_paths[
    (all_paths['min_weight'] > 30) &                    # Strong connections
    (all_paths['traversal_probability'] > 0.6) &        # High probability
    (all_paths['inter_layer_num'] == 2) &               # Exactly 2 intermediate hops
    (all_paths['path_block'].str.contains('Mi1|Mi4'))   # Through Mi1 OR Mi4
]

print(f"Advanced filtered paths: {len(advanced_filter)}")

vp = VisualizePath(
    path_file=advanced_filter,
    output_folder='./advanced_filtered',
    network_layout='spring',
    showfig=False
)

conn_df, G = vp.visualize()

print("\n" + "="*80)
print("All examples complete!")
print("="*80)
print("\nSummary:")
print("  • VisualizePath class: Standalone usage, most flexible")
print("  • visualize_paths(): Quick one-liner function")
print("  • All methods produce: Sankey HTML + Network HTML + Data Excel")
print("\nRecommendation:")
print("  → Use standalone VisualizePath for post-analysis visualization")
print("  → No need to initialize FindNeuronConnection!")
print("\nOutput files created:")
print("  • sankey_selected_paths.html - Flow-based diagram")
print("  • network_selected_paths.html - Interactive network (drag, hide, hover)")
print("  • selected_paths_connections.xlsx - Connection data")
