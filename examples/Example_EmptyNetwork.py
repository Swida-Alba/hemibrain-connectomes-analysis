#!/usr/bin/env python3
"""
Example: Generate Empty Network HTML Template

This example demonstrates how to generate an empty network visualization
template using the VisualizePath class.

The empty network includes:
- Full Cytoscape.js interface
- All interactive controls (layout, zoom, filters)
- Export functionality
- No predefined nodes or edges
- Unique timestamped filename to prevent overwrites

Useful for:
- Creating network templates
- Testing visualization interface
- Building custom network visualizations
- Prototyping network layouts

Author: Kun-Da Wu
Date: 2025-11-08
"""

import sys
from pathlib import Path
import time

# Add parent directory to path to import vispath_pkg
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / 'vispath-subproject' / 'src'))

from vispath_pkg import VisualizePath

# =============================================================================
# Generate Empty Network Template
# =============================================================================

print("="*80)
print("Example: Generate Empty Network HTML Template")
print("="*80)

# Create VisualizePath instance with generate_empty_network=True
# No path_file required when this flag is set
vp = VisualizePath(
    path_file=None,  # Not needed for empty network
    output_folder='./empty_network_example',
    network_layout='hierarchical',
    generate_empty_network=True,  # Enable empty network generation
    showfig=True  # Open in browser automatically
)

# Generate the empty network HTML
# Filename will include timestamp: empty_network_example_YYYYMMDD_HHMMSS_network.html
vp.visualize()

print("\n" + "="*80)
print("Empty network HTML generated!")
print("="*80)
print("\nThe generated HTML includes:")
print("  • Full Cytoscape.js network interface")
print("  • Interactive controls (drag, zoom, hide)")
print("  • Layout algorithms (hierarchical, spring, etc.)")
print("  • Export to PNG functionality")
print("  • No predefined nodes or edges (blank canvas)")
print("  • Unique timestamp in filename to prevent overwrites")
print("\nYou can:")
print("  1. Use it as a template for custom visualizations")
print("  2. Add nodes/edges programmatically via JavaScript")
print("  3. Test the network interface without data")
print("="*80)

# =============================================================================
# Generate Multiple Empty Networks - Each Gets Unique Filename
# =============================================================================

print("\n" + "="*80)
print("Generating multiple empty networks (each with unique timestamp)")
print("="*80)

for i in range(3):
    vp_multi = VisualizePath(
        path_file=None,
        output_folder='./empty_network_multiple',
        generate_empty_network=True,
        showfig=False  # Don't open all of them
    )
    
    output_path = vp_multi.generate_empty_network_html()
    print(f"  Generated #{i+1}: {output_path}")
    
    # Small delay to ensure different timestamps
    time.sleep(1)

print("\n✓ Generated 3 unique empty network files")
print("  Each has a different timestamp in the filename")
print("="*80)

# =============================================================================
# Alternative: Using direct method call
# =============================================================================

print("\n" + "="*80)
print("Alternative: Using direct method call")
print("="*80)

vp2 = VisualizePath(
    path_file=None,
    output_folder='./empty_network_direct',
    generate_empty_network=True,
    showfig=False  # Don't open automatically this time
)

# Call the method directly instead of visualize()
output_path = vp2.generate_empty_network_html()

print(f"\n✓ Empty network saved to: {output_path}")
print("  Filename includes timestamp for uniqueness")
print("="*80)
