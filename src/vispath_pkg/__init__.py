"""
VisualizePath Package

A standalone visualization toolkit for neural pathways.

This package is fully standalone and includes all necessary visualization
functions including heatmaps, Sankey diagrams, and network graphs.
"""

# Import the main VisualizePath class and visualization functions
from .vispath import VisualizePath, parse_color_to_hex_opacity, VisConnMatInteractive, visualize_paths

__version__ = "1.0.0"
__author__ = "Kun-Da Wu"

__all__ = ['VisualizePath', 'parse_color_to_hex_opacity', 'VisConnMatInteractive', 'visualize_paths']

