"""Shared options for analysis-generated visualizations.

This module intentionally stays lightweight so analysis backends can resolve
dataset-aware rendering defaults without importing the full visualization
stack.
"""


def default_analysis_skeleton_mesh_simplification(
        dataset: str, neuprint_skeleton_pipeline: str = "fine") -> float:
    """Return the default tube-mesh simplification for analysis renders.

    Analysis tabs render result collections, so fine/artistic use the more
    aggressive 98% setting.  The direct/fast NeuPrint path keeps its fixed
    simp90 convention and therefore defaults to 90% removal.  The dedicated
    Skeleton tab keeps its separate 95% fine/artistic default.
    """

    pipeline = str(neuprint_skeleton_pipeline or "fine").strip().lower()
    if pipeline in {"fast", "direct"}:
        return 0.90
    return 0.98
