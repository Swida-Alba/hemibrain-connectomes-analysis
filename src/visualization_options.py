"""Shared options for analysis-generated visualizations.

This module intentionally stays lightweight so analysis backends can resolve
dataset-aware rendering defaults without importing the full visualization
stack.
"""


def _is_flywire_family(dataset: str) -> bool:
    """Return whether *dataset* uses the FlyWire/FAFB render family."""
    normalized = str(dataset or "").strip().lower()
    return normalized.startswith("flywire_") or "fafb" in normalized


def default_skeleton_tab_simplification(
        dataset: str, neuprint_skeleton_pipeline: str = "fast") -> float:
    """Return the default target for the dedicated Skeleton tab.

    FAFB/FlyWire meshes use the dedicated 95% target regardless of the
    disabled NeuPrint method selector shown in the UI. NeuPrint keeps the
    method-specific direct/fast 90% target and fine/artistic 95% target.
    """
    if _is_flywire_family(dataset):
        return 0.95
    pipeline = str(neuprint_skeleton_pipeline or "fast").strip().lower()
    return 0.90 if pipeline in {"fast", "direct"} else 0.95


def default_analysis_skeleton_mesh_simplification(
        dataset: str, neuprint_skeleton_pipeline: str = "fine") -> float:
    """Return the default tube-mesh simplification for analysis renders.

    Analysis tabs render result collections, so fine/artistic use the more
    aggressive 98% setting.  FAFB analysis renders use 98% regardless of the
    disabled NeuPrint selector.  The direct/fast NeuPrint path keeps its
    fixed simp90 convention and therefore defaults to 90% removal.  The
    dedicated Skeleton tab keeps its separate 95% fine/artistic default.
    """

    # Similarity and NeuronBridge renders are sub-called analysis workflows;
    # FAFB intentionally uses a coarser 98% render target there so result
    # collections remain responsive.
    if _is_flywire_family(dataset):
        return 0.98

    pipeline = str(neuprint_skeleton_pipeline or "fine").strip().lower()
    if pipeline in {"fast", "direct"}:
        return 0.90
    return 0.98
