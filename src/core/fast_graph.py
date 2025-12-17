"""
FastGraph - Lightweight Graph Implementation for Connectome Analysis

This module re-exports FastGraph from vispath_pkg to maintain a single source
of truth while preserving backward compatibility for existing imports.

The full implementation is in vispath-subproject/src/vispath_pkg/fast_graph.py
which provides:
- NetworkX-compatible API with node/edge attributes
- All pathfinding algorithms (DFS, BFS, meet-in-the-middle, etc.)
- DataFrame construction methods for both Pandas and Polars
- Label aggregation for type-level analysis

Example Usage:
    from core.fast_graph import FastGraph
    
    # Build graph from DataFrame
    G = FastGraph.build_from_dataframe(
        df, source_col='pre', target_col='post', weight_col='weight'
    )
    
    # Find all paths
    for path in G.all_simple_paths(source, target, cutoff=3):
        print(path)
    
    # Aggregate by neuron type
    G_type = G.aggregate_by_label(bodyid_to_type_map)
"""

import sys
from pathlib import Path

# Add vispath-subproject to path for import
_vispath_path = Path(__file__).parent.parent.parent / "vispath-subproject" / "src"
if str(_vispath_path) not in sys.path:
    sys.path.insert(0, str(_vispath_path))

# Import and re-export FastGraph from vispath_pkg
from vispath_pkg.fast_graph_core import FastGraph, DiGraph

__all__ = ['FastGraph', 'DiGraph']
