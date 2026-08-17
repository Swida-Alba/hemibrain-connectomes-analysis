"""Shared options for analysis-generated visualizations.

This module intentionally stays lightweight so analysis backends can resolve
dataset-aware rendering defaults without importing the full visualization
stack.
"""


def default_analysis_skeleton_mesh_simplification(dataset: str) -> float:
    """Return the default tube-mesh simplification for analysis renders.

    Analysis tabs render result collections, so they use the more aggressive
    98% setting.  The dedicated Skeleton tab keeps its separate 95% default.
    """

    return 0.98
