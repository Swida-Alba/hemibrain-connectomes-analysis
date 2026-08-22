"""
DROCAT - Core Module

This module provides tools for analyzing and visualizing Drosophila hemibrain connectome data.

Main components:
- coana: Connectome analysis (FindNeuronConnection)
- statvis: Statistical visualization and analysis
- vispath_pkg: Pathway visualization (imported from vispath-subproject)
- comparison: Cross-dataset comparison tools
"""

import sys
from pathlib import Path

# Add vispath-subproject to path to make vispath_pkg importable
vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

# Make vispath_pkg available when importing from src
try:
    from vispath_pkg import VisualizePath, visualize_paths, parse_color_to_hex_opacity, VisConnMatInteractive
    __all__ = ['VisualizePath', 'visualize_paths', 'parse_color_to_hex_opacity', 'VisConnMatInteractive']
except ImportError:
    __all__ = []

# Import comparison module components
try:
    from .comparison import (
        ComparisonAnalyzer,
        DatasetConfig,
        ComparisonParameters,
        LabelMapper,
        DataLoader,
        ComparisonMetrics,
        ComparisonVisualizer,
        quick_compare
    )
    __all__.extend([
        'ComparisonAnalyzer',
        'DatasetConfig', 
        'ComparisonParameters',
        'LabelMapper',
        'DataLoader',
        'ComparisonMetrics',
        'ComparisonVisualizer',
        'quick_compare'
    ])
except ImportError:
    pass