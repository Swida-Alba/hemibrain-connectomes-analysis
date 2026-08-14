"""Shared options for analysis-generated visualizations.

This module intentionally stays lightweight so analysis backends can resolve
dataset-aware rendering defaults without importing the full visualization
stack.
"""


def default_analysis_skeleton_mesh_simplification(dataset: str) -> float:
    """Return the default tube-mesh simplification for analysis renders.

    Analysis tabs render result collections, so they use a little more
    simplification than the dedicated Skeleton tab's historical defaults.
    FlyWire FAFB meshes are larger and receive the more aggressive setting.
    """

    dataset_name = str(dataset or "").lower()
    if "fafb" in dataset_name or "flywire_fafb" in dataset_name:
        return 0.98
    return 0.95
