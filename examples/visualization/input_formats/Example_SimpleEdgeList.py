#!/usr/bin/env python3
"""
Example: Using VisualizePath with Flexible Edge-List Format

This example demonstrates the enhanced edge-list format with flexible column
recognition and automatic metric detection.

Minimum requirements: 3 columns (source + target + weight)

Supported column names:
  Source:  'source', 'from', 'pre', or any '*_pre' (e.g., bodyId_pre, type_pre)
  Target:  'target', 'to', 'post', or any '*_post' (e.g., bodyId_post, type_post)
  Weight:  'weight', 'weights', 'synapse_count', 'count'

Additional features:
  - Automatic detection of numeric columns as additional metrics
  - Ratio/probability columns auto-mapped for toggle support
  - Custom metric columns preserved for data export
  - All formats work with CSV, Excel, or DataFrame input
"""

import sys
from pathlib import Path

# Add vispath-subproject to Python path for local development
vispath_pkg_path = Path(__file__).parent.parent.parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

import pandas as pd

from vispath_pkg import VisualizePath

print("="*80)
print("VisualizePath - Flexible Edge-List Format Examples")
print("="*80)
print()

# ==============================================================================
# Example 1: Simple source/target/weight format
# ==============================================================================
print("Example 1: Simple edge-list with source/target/weight columns")
print("-" * 70)

# Create simple edge data
edges_data = {
    'source': ['A', 'A', 'B', 'B', 'C', 'D'],
    'target': ['B', 'C', 'C', 'D', 'E', 'E'],
    'weight': [10, 5, 8, 12, 6, 15]
}

df_edges = pd.DataFrame(edges_data)
print("\nInput data:")
print(df_edges)
print()

# Visualize
vis1 = VisualizePath(
    path_file=df_edges,
    output_folder='./output_example1_simple',
    source_color='#3498db',
    target_color='#e74c3c',
    intermediate_color='#2ecc71'
)

# Create visualizations
vis1.create_network()
vis1.create_sankey()

print("\n✓ Visualizations created in: ./output_example1_simple/")
print("  - network_selected_paths.html")
print("  - sankey_selected_paths.html")
print()

# ==============================================================================
# Example 2: BodyId format (bodyId_pre/bodyId_post/weight)
# ==============================================================================
print("Example 2: BodyId edge-list format")
print("-" * 70)

# Create bodyId edge data
bodyid_data = {
    'bodyId_pre': [123456, 123456, 234567, 234567, 345678],
    'bodyId_post': [234567, 345678, 345678, 456789, 567890],
    'weight': [25, 15, 30, 20, 18]
}

df_bodyid = pd.DataFrame(bodyid_data)
print("\nInput data:")
print(df_bodyid)
print()

# Visualize
vis2 = VisualizePath(
    path_file=df_bodyid,
    output_folder='./output_example2_bodyid',
    source_color='rgba(52, 152, 219, 0.8)',
    target_color='rgba(231, 76, 60, 0.8)'
)

vis2.create_network()
vis2.create_sankey()

print("\n✓ Visualizations created in: ./output_example2_bodyid/")
print()

# ==============================================================================
# Example 3: from/to format
# ==============================================================================
print("Example 3: From/To edge-list format")
print("-" * 80)

# Create from/to edge data
fromto_data = {
    'from': ['PN1', 'PN1', 'PN2', 'LN1', 'LN1', 'LN2'],
    'to': ['LN1', 'LN2', 'LN1', 'MBON1', 'MBON2', 'MBON2'],
    'weight': [50, 30, 40, 25, 35, 28]
}

df_fromto = pd.DataFrame(fromto_data)
print("\nInput data:")
print(df_fromto)
print()

# Visualize
vis3 = VisualizePath(
    path_file=df_fromto,
    output_folder='./output_example3_fromto',
    link_color='rgba(100, 100, 100, 0.4)'
)

vis3.create_network()
vis3.create_sankey()

print("\n✓ Visualizations created in: ./output_example3_fromto/")
print()

# ==============================================================================
# Example 4: Edge-list with ratio and probability columns
# ==============================================================================
print("Example 4: Edge-list with additional metric columns")
print("-" * 80)

# Create data with multiple numeric columns
# VisualizePath will automatically detect ratio and probability columns
metrics_data = {
    'source': ['KC_a', 'KC_a', 'KC_b', 'KC_b', 'MBON_a'],
    'target': ['MBON_a', 'MBON_b', 'MBON_a', 'MBON_b', 'DAN'],
    'weight': [100, 50, 80, 60, 45],
    'ratio': [0.67, 0.33, 0.57, 0.43, 1.0],  # Auto-mapped to connection_ratios
    'probability': [0.95, 0.85, 0.92, 0.88, 0.98]  # Auto-mapped to traversal_probabilities
}

df_metrics = pd.DataFrame(metrics_data)
print("\nInput data with additional metrics:")
print(df_metrics)
print()
print("Note: 'ratio' → connection_ratios, 'probability' → traversal_probabilities")
print("      These metrics can be toggled in the interactive visualizations")
print()

# Visualize
vis4 = VisualizePath(
    path_file=df_metrics,
    output_folder='./output_example4_metrics'
)

vis4.create_network()
vis4.create_sankey()
vis4.create_heatmap()

print("\n✓ Visualizations created with toggleable metrics in: ./output_example4_metrics/")
print()

# ==============================================================================
# Example 5: Edge-list with custom metric columns
# ==============================================================================
print("Example 5: Edge-list with custom named metrics")
print("-" * 80)

# Create data with custom metric columns
# All numeric columns (except source, target, color) are detected as metrics
custom_data = {
    'neuron_pre': ['DA1_PN', 'DA1_PN', 'VA1d_PN', 'VA1d_PN'],
    'neuron_post': ['LHON1', 'LHON2', 'LHON1', 'LHON3'],
    'synapse_count': [150, 120, 180, 90],
    'strength': [0.85, 0.72, 0.91, 0.68],
    'confidence': [0.95, 0.88, 0.92, 0.85]
}

df_custom = pd.DataFrame(custom_data)
print("\nInput data with custom metrics:")
print(df_custom)
print()
print("Detected numeric columns: synapse_count, strength, confidence")
print("All metrics are preserved for data export")
print()

# Visualize
vis5 = VisualizePath(
    path_file=df_custom,
    output_folder='./output_example5_custom'
)

vis5.create_network()
vis5.create_sankey()

print("\n✓ Visualizations created with custom metrics in: ./output_example5_custom/")
print()

# ==============================================================================
# Example 6: Reading from CSV file with metrics
# ==============================================================================
print("Example 6: Loading edge-list from CSV file")
print("-" * 80)

# Create CSV file with metrics
csv_file = './test_edges_metrics.csv'
edges_csv = pd.DataFrame({
    'source': ['KC_a', 'KC_a', 'KC_b', 'KC_b', 'MBON_a'],
    'target': ['MBON_a', 'MBON_b', 'MBON_a', 'MBON_b', 'DAN'],
    'weight': [100, 50, 80, 60, 45],
    'ratio': [0.75, 0.38, 0.62, 0.46, 1.0]
})
edges_csv.to_csv(csv_file, index=False)
print(f"\nCreated CSV file: {csv_file}")
print(edges_csv)
print()

# Load and visualize from CSV
vis6 = VisualizePath(
    path_file=csv_file,
    output_folder='./output_example6_csv'
)

vis6.create_network()
vis6.create_sankey()

print("\n✓ Visualizations created in: ./output_example6_csv/")
print()

# ==============================================================================
# Example 7: Reading from Excel file with multiple metrics
# ==============================================================================
print("Example 7: Loading edge-list from Excel file")
print("-" * 80)

# Create Excel file with multiple metrics
excel_file = './test_edges_multi.xlsx'
edges_excel = pd.DataFrame({
    'neuron_pre': ['DA1_PN', 'DA1_PN', 'VA1d_PN', 'VA1d_PN'],
    'neuron_post': ['LHON1', 'LHON2', 'LHON1', 'LHON3'],
    'synapse_count': [150, 120, 180, 90],
    'probability': [0.92, 0.88, 0.95, 0.85],
    'reliability': [0.87, 0.82, 0.91, 0.79]
})
edges_excel.to_excel(excel_file, index=False)
print(f"\nCreated Excel file: {excel_file}")
print(edges_excel)
print()
print("Note: probability → traversal_probabilities (toggleable)")
print("      reliability → custom metric (preserved for export)")
print()

# Load and visualize from Excel
vis7 = VisualizePath(
    path_file=excel_file,
    output_folder='./output_example7_excel'
)

vis7.create_network()
vis7.create_sankey()

print("\n✓ Visualizations created in: ./output_example7_excel/")
print()

# ==============================================================================
# Summary
# ==============================================================================
print("="*80)
print("Summary: Flexible Edge-List Format")
print("="*80)
print()
print("✨ Minimum requirements: 3 columns (source + target + weight)")
print()
print("📋 Column name options:")
print("  Source:  'source', 'from', 'pre', or any '*_pre' (e.g., bodyId_pre, type_pre)")
print("  Target:  'target', 'to', 'post', or any '*_post' (e.g., bodyId_post, type_post)")
print("  Weight:  'weight', 'weights', 'synapse_count', 'count'")
print()
print("🎯 Basic format examples:")
print("  1. source | target | weight")
print("  2. from | to | weight")
print("  3. pre | post | weight")
print("  4. bodyId_pre | bodyId_post | weight")
print("  5. neuron_pre | neuron_post | synapse_count")
print()
print("📊 Additional metric columns (automatically detected):")
print("  • All numeric columns (except source, target, color) are detected")
print("  • 'ratio' → auto-mapped to connection_ratios (toggleable)")
print("  • 'probability' → auto-mapped to traversal_probabilities (toggleable)")
print("  • Custom numeric columns preserved for data export")
print()
print("💾 Supported input formats:")
print("  • pandas DataFrame")
print("  • CSV file (.csv)")
print("  • Excel file (.xlsx, .xls)")
print()
print("🎨 Features with metrics:")
print("  • Toggle metrics on/off in interactive visualizations")
print("  • Metrics affect edge width, color intensity, and labels")
print("  • All metrics exported in saved data files")
print()
print("The system automatically:")
print("  ✓ Detects column names (flexible matching)")
print("  ✓ Identifies all numeric columns as metrics")
print("  ✓ Maps common metric names to standard columns")
print("  ✓ Preserves custom metrics for analysis")
print("  ✓ Creates interactive visualizations with toggle controls")
print()
print("="*80)
