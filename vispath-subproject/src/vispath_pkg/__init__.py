"""
VisualizePath Package

A standalone visualization toolkit for neural pathways.

This package is fully standalone and includes all necessary visualization
functions including heatmaps, Sankey diagrams, and network graphs.
"""

# Import the lightweight graph implementation
from .fast_graph_core import FastGraph

# Import the main VisualizePath class and visualization functions
from .vispath import (
    VisualizePath,
    parse_color_to_hex_opacity,
    VisConnMatInteractive,
    visualize_paths,
    visualize_heatmap,
    visualize_sankey,
    visualize_network
)

__version__ = "1.0.0"
__author__ = "Kun-Da Wu"

__all__ = [
    'FastGraph',
    'VisualizePath',
    'parse_color_to_hex_opacity',
    'VisConnMatInteractive',
    'visualize_paths',
    'visualize_heatmap',
    'visualize_sankey',
    'visualize_network'
]

