"""
Comprehensive test for VisualizePath input formats and visualization options.

Tests:
1. Connection matrix input (square and rectangular)
2. Edge-list input (various column name formats)
3. Path-based input
4. Individual visualization control (plot_heatmap, plot_Sankey, plot_network)
"""

import sys
from pathlib import Path

# Add vispath-subproject to path (visualization/input_formats/ -> visualization/ -> examples/ -> project root)
vispath_path = Path(__file__).parent.parent.parent.parent / 'vispath-subproject' / 'src'
sys.path.insert(0, str(vispath_path))

import numpy as np
import pandas as pd
from vispath_pkg.vispath import VisualizePath
import os

print("=" * 80)
print("VisualizePath - Comprehensive Input Format Test")
print("=" * 80)

# Test 1: Connection Matrix Input (Square)
print("\n--- Test 1: Square Connection Matrix (5x5) ---")
matrix_square = pd.DataFrame(
    np.random.poisson(3, (5, 5)),
    index=['A', 'B', 'C', 'D', 'E'],
    columns=['A', 'B', 'C', 'D', 'E']
)
vp1 = VisualizePath(matrix_square, output_folder='./test_output/matrix_square')
vp1.visualize(plot_heatmap=True, plot_Sankey=False, plot_network=False)
print("✓ Test 1 passed: Square matrix with heatmap only")

# Test 2: Connection Matrix Input (Rectangular)
print("\n--- Test 2: Rectangular Connection Matrix (10x12) ---")
matrix_rect = pd.DataFrame(
    np.random.poisson(2, (10, 12)),
    index=[f"Source_{i}" for i in range(10)],
    columns=[f"Target_{j}" for j in range(12)]
)
vp2 = VisualizePath(matrix_rect, output_folder='./test_output/matrix_rect')
vp2.visualize(plot_heatmap=False, plot_Sankey=True, plot_network=False)
print("✓ Test 2 passed: Rectangular matrix with Sankey only")

# Test 3: Connection Matrix without Index/Columns (auto-generation)
print("\n--- Test 3: Matrix without Named Index/Columns ---")
matrix_numeric = pd.DataFrame(np.random.poisson(2, (6, 6)))
vp3 = VisualizePath(matrix_numeric, output_folder='./test_output/matrix_numeric')
vp3.visualize(plot_heatmap=False, plot_Sankey=False, plot_network=True)
print("✓ Test 3 passed: Numeric matrix with network only")

# Test 4: Edge-list Input (Standard format: source, target, weight)
print("\n--- Test 4: Edge-list (source, target, weight) ---")
edge_df1 = pd.DataFrame({
    'source': ['A', 'B', 'C', 'D'],
    'target': ['B', 'C', 'D', 'E'],
    'weight': [10, 15, 8, 12]
})
vp4 = VisualizePath(edge_df1, output_folder='./test_output/edge_standard')
vp4.visualize()  # All visualizations
print("✓ Test 4 passed: Standard edge-list with all visualizations")

# Test 5: Edge-list Input (Alternative format: from, to, weight)
print("\n--- Test 5: Edge-list (from, to, weight) ---")
edge_df2 = pd.DataFrame({
    'from': ['A', 'B', 'C'],
    'to': ['B', 'C', 'D'],
    'weight': [10, 15, 8]
})
vp5 = VisualizePath(edge_df2, output_folder='./test_output/edge_fromto')
vp5.visualize(plot_heatmap=True, plot_Sankey=True, plot_network=False)
print("✓ Test 5 passed: from/to edge-list with heatmap and Sankey")

# Test 6: Edge-list Input (Prefixed format: bodyId_pre, bodyId_post)
print("\n--- Test 6: Edge-list (bodyId_pre, bodyId_post, weight) ---")
edge_df3 = pd.DataFrame({
    'bodyId_pre': [123, 456, 789, 123],
    'bodyId_post': [456, 789, 123, 789],
    'weight': [10, 15, 8, 20]
})
vp6 = VisualizePath(edge_df3, output_folder='./test_output/edge_prefixed')
vp6.visualize(plot_heatmap=False, plot_Sankey=True, plot_network=True)
print("✓ Test 6 passed: Prefixed edge-list with Sankey and network")

# Test 7: Path-based Input
print("\n--- Test 7: Path-based format ---")
path_df = pd.DataFrame({
    'path_block': [
        'A -> B -> C',
        'A -> D -> C',
        'B -> E -> F'
    ],
    'weights': [
        [10, 5],
        [15, 8],
        [12, 6]
    ],
    'connection_ratios': [
        [0.5, 0.3],
        [0.6, 0.4],
        [0.55, 0.35]
    ]
})
vp7 = VisualizePath(path_df, output_folder='./test_output/path_based')
vp7.visualize()
print("✓ Test 7 passed: Path-based input with all visualizations")

# Test 8: All visualizations disabled (should still work, just no visualizations generated)
print("\n--- Test 8: No visualizations (data processing only) ---")
vp8 = VisualizePath(edge_df1, output_folder='./test_output/no_vis')
vp8.visualize(plot_heatmap=False, plot_Sankey=False, plot_network=False)
print("✓ Test 8 passed: Data processing without visualizations")

print("\n" + "=" * 80)
print("✓ All tests passed successfully!")
print("=" * 80)
print("\nOutput files generated in ./test_output/")
print("Check each subfolder for the respective visualization outputs.")
