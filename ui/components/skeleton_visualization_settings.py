"""Shared controls for background skeleton visualizations.

The standalone 3D Skeleton tab exposes the complete editor because it is a
dedicated visualization workflow.  The analysis tabs need the same rendering
choices without duplicating a large collection of inputs in every tab, so
they use :func:`skeleton_visualization_settings` in a collapsed expansion.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from nicegui import ui

from ..config import BRAIN_MESH_OPTIONS, SKELETON_MODES
from .common import (
    checkbox_input,
    multi_select_input,
    number_input,
    param_grid,
    select_input,
)
from .palette_picker import color_swatch_picker, palette_editor


def _default_analysis_simplification(dataset_value: Any) -> float:
    """Return the visible analysis default for one or more datasets."""

    if isinstance(dataset_value, (list, tuple, set)):
        dataset_names = dataset_value
    else:
        dataset_names = [dataset_value]
    normalized = [str(name or "").lower() for name in dataset_names]
    if any("fafb" in name or "flywire_fafb" in name for name in normalized):
        return 0.98
    return 0.95


@dataclass
class SkeletonVisualizationSettings:
    """Live NiceGUI controls plus a normalized value snapshot method."""

    fields: Dict[str, Any]
    panel: Any = None

    def values(self) -> Dict[str, Any]:
        """Return values using the keyword names accepted by VisualizeSkeleton."""
        values: Dict[str, Any] = {}

        def palette_value(field):
            """Return a subprocess-safe value while retaining continuous metadata."""
            colors = field.get_colors()
            if getattr(colors, "is_continuous_palette", False):
                return {
                    "colors": list(colors),
                    "continuous": True,
                }
            return colors

        for name, field in self.fields.items():
            if name in {"neuron_colors", "synapse_colors", "roi_colors"}:
                values[name] = palette_value(field)
            elif name in {"brain_mesh_color", "vnc_mesh_color"}:
                values[name] = field.get_value()
            else:
                values[name] = field.value

        if "visualize_top_n" in values:
            values["visualize_top_n"] = int(values["visualize_top_n"] or 0)
        if "export_scale" in values:
            values["export_scale"] = int(values["export_scale"] or 1)
        if "min_synapse_num" in values:
            values["min_synapse_num"] = int(values["min_synapse_num"] or 1)
        if "neuron_alpha" in values:
            values["neuron_alpha"] = float(values["neuron_alpha"] or 0)
        if "synapse_alpha" in values:
            values["synapse_alpha"] = float(values["synapse_alpha"] or 0)
        if "mesh_alpha" in values:
            values["mesh_alpha"] = float(values["mesh_alpha"] or 0)
        if "skeleton_mesh_simplification" in values:
            if values.get("use_default_simplification", True):
                values["skeleton_mesh_simplification"] = None
            else:
                values["skeleton_mesh_simplification"] = float(
                    values["skeleton_mesh_simplification"] or 0
                )
            values.pop("use_default_simplification", None)
        if "mesh_roi" in values:
            values["mesh_roi"] = list(values["mesh_roi"] or [])
        if "roi_colors" in values:
            values["mesh_color"] = values.pop("roi_colors")
        return values

    def warn_empty_custom_palettes(self) -> None:
        """Warn when a custom palette is empty (backend default fallback).

        The ROI palette is only consulted when at least one ROI mesh is
        selected, so it is skipped otherwise.
        """
        from .palette_picker import notify_empty_custom_palettes

        palettes = [
            (self.fields["neuron_colors"], "Neuron Colors"),
            (self.fields["synapse_colors"], "Synapse Colors"),
        ]
        if self.fields["mesh_roi"].value:
            palettes.append((self.fields["roi_colors"], "ROI Colors"))
        notify_empty_custom_palettes(*palettes)


def skeleton_visualization_settings(
    *,
    default_top_n: int = 5,
    top_n_label: str = "Visualize Top N",
    top_n_hint: str = "Number of ranked results to render as 3D skeletons.",
    default_visualize_by: str = "type",
    include_ranking: bool = True,
    default_brain_mesh: str = "template",
    default_neuron_alpha: float = 0.3,
    default_show_fig: bool = False,
    default_export_views: bool = False,
    default_export_method: str = "webdriver",
    dataset_provider: Optional[Callable[[], Any]] = None,
    dataset_watchers: Optional[Iterable[Any]] = None,
) -> SkeletonVisualizationSettings:
    """Create the collapsed advanced visualization editor used by analysis tabs.

    The returned ``values()`` dictionary contains the common, user-facing
    ``VisualizeSkeleton`` settings: grouping, mesh selection, appearance,
    synapses, ROI mesh selection, caching, simplification, and export behavior.
    ``include_ranking`` adds the tab-specific top-N and type/bodyId controls.
    """
    fields: Dict[str, Any] = {}

    with ui.expansion(
        "Advanced Visualization",
        icon="view_in_ar",
    ).classes("flex-grow drocat-advanced-visualization") as panel:
        ui.label(
            "These settings apply only to the optional skeleton visualizations "
            "generated by this analysis."
        ).classes("text-caption drocat-muted")

        if include_ranking:
            with param_grid(2):
                fields["visualize_top_n"] = number_input(
                    top_n_label,
                    default_top_n,
                    1,
                    100,
                    hint=top_n_hint,
                )
                fields["visualize_by"] = select_input(
                    "Visualize By",
                    ["type", "bodyId"],
                    default_visualize_by,
                    hint=(
                        "'type': group member neurons by type. "
                        "'bodyId': show individual neurons."
                    ),
                )

        ui.label("Appearance").classes("drocat-mini-label")
        with param_grid(3):
            fields["skeleton_mode"] = select_input(
                "Skeleton Mode",
                SKELETON_MODES,
                "tube",
                hint="'tube' is detailed; 'line' is faster for many neurons.",
            )
            fields["legend_mode"] = select_input(
                "Legend Mode",
                ["layer", "type", "single"],
                "layer",
                hint="Choose one legend entry per layer, type, or individual neuron.",
            )
            fields["background_color"] = select_input(
                "Background",
                ["white", "black"],
                "white",
                hint="Background color for the interactive scene and exported views.",
            )
            fields["brain_mesh"] = select_input(
                "Brain Mesh",
                BRAIN_MESH_OPTIONS,
                default_brain_mesh,
                hint=(
                    "'template': dataset-aligned template brain. 'whole': full "
                    "standard surface. 'none': no brain mesh."
                ),
            )
            fields["vnc_mesh"] = checkbox_input(
                "VNC Mesh",
                False,
                hint="Show the ventral nerve cord mesh when the dataset supports it.",
            )

        with ui.row().classes("w-full items-start gap-4"):
            with ui.column().classes("flex-grow"):
                fields["neuron_colors"] = palette_editor(
                    "Neuron Colors",
                    value="Category10",
                    include_auto=False,
                )
                fields["neuron_alpha"] = number_input(
                    "Neuron Opacity",
                    default_neuron_alpha,
                    0,
                    1,
                    0.1,
                    hint=(
                        "Global fallback opacity for skeletons (0=invisible, 1=solid). "
                        "A color with an explicit opacity channel (#RGBA/#RRGGBBAA, "
                        "rgba(), or an RGBA tuple) overrides this value for that "
                        "layer; colors without opacity inherit it."
                    ),
                ).classes("w-48")
            fields["brain_mesh_color"] = color_swatch_picker(
                "Brain Mesh Color",
                value="auto",
            ).classes("flex-grow")
            fields["vnc_mesh_color"] = color_swatch_picker(
                "VNC Mesh Color",
                value="auto",
            ).classes("flex-grow")

        ui.label("Synapses and regions").classes("drocat-mini-label")
        with param_grid(3):
            fields["skip_synapse"] = checkbox_input(
                "Skip Synapses",
                True,
                hint="Hide synapse markers for a cleaner skeleton view.",
            )
            fields["min_synapse_num"] = number_input(
                "Min Synapse Count",
                3,
                1,
                1000,
                hint="Minimum synapses required for a connection marker.",
            )
            fields["synapse_mode"] = select_input(
                "Synapse Mode",
                ["cone", "scatter"],
                "cone",
                hint="Directional cones or simple point markers.",
            )
            fields["synapse_size"] = select_input(
                "Synapse Size",
                ["real", "1", "2", "3"],
                "real",
                hint="Use real count scaling or a fixed marker size.",
            )
            fields["synapse_alpha"] = number_input(
                "Synapse Opacity",
                0.6,
                0,
                1,
                0.1,
                hint=(
                    "Global fallback opacity for synapse markers. A color with an "
                    "explicit opacity channel overrides this value; colors "
                    "without opacity inherit it."
                ),
            )
            fields["mesh_alpha"] = number_input(
                "ROI Mesh Opacity",
                0.1,
                0,
                1,
                0.05,
                hint=(
                    "Global fallback opacity for ROI meshes. A color with an "
                    "explicit opacity channel overrides this value; colors "
                    "without opacity inherit it."
                ),
            )
        fields["synapse_colors"] = palette_editor(
            "Synapse Colors",
            value="Dark2",
            include_auto=False,
        )
        fields["mesh_roi"] = multi_select_input(
            "Mesh ROIs (optional)",
            [],
            default=[],
            hint="Type ROI names and press Enter to add optional region meshes.",
        ).props("outlined").props('new-value-mode="add-unique"')
        fields["roi_colors"] = palette_editor(
            "ROI Colors",
            value="Cool",
            include_auto=True,
        )

        ui.label("Data and export").classes("drocat-mini-label")
        with param_grid(3):
            fields["cache_neurons"] = checkbox_input(
                "Cache Neurons",
                True,
                hint="Cache fetched skeletons for faster repeat renders.",
            )
            fields["cache_synapses"] = checkbox_input(
                "Cache Synapses",
                True,
                hint="Cache fetched synapse data for faster repeat renders.",
            )
            fields["smooth_skeleton"] = checkbox_input(
                "Smooth Skeleton",
                False,
                hint="Apply smoothing to skeleton tube meshes.",
            )
            fields["show_soma"] = checkbox_input(
                "Show Soma",
                True,
                hint="Render soma spheres when available.",
            )
            fields["show_connectors"] = checkbox_input(
                "Show Connectors",
                False,
                hint="Render synaptic connector markers.",
            )
            fields["use_default_simplification"] = checkbox_input(
                "Default Simplification",
                True,
                hint=(
                    "Use the analysis default: 0.95 for NeuPrint datasets or "
                    "0.98 for FlyWire FAFB."
                ),
            )
            fields["skeleton_mesh_simplification"] = number_input(
                "Mesh Simplification",
                _default_analysis_simplification(
                    dataset_provider() if dataset_provider else None
                ),
                0,
                0.99,
                0.05,
                hint=(
                    "Fraction of skeleton-mesh faces removed (higher is coarser). "
                    "The displayed default follows the selected dataset: 0.95 "
                    "for NeuPrint or 0.98 for FlyWire FAFB."
                ),
            )
            fields["export_method"] = select_input(
                "Export Method",
                ["webdriver", "kaleido"],
                default_export_method,
                hint="Renderer used for PNG exports.",
            )
            fields["export_scale"] = number_input(
                "Export Scale",
                3,
                1,
                10,
                hint="Resolution multiplier for exported images.",
            )
            fields["show_fig"] = checkbox_input(
                "Show Figure",
                default_show_fig,
                hint="Open the interactive figure after rendering.",
            )
            fields["export_views"] = checkbox_input(
                "Export Views",
                default_export_views,
                hint="Export the configured view images after rendering.",
            )

        # The simplification input is only meaningful when the default is
        # disabled.  Keep the value in the returned dictionary regardless so
        # callers can take one consistent snapshot at run time.
        def refresh_default_simplification(_event=None):
            if bool(fields["use_default_simplification"].value):
                fields["skeleton_mesh_simplification"].set_value(
                    _default_analysis_simplification(
                        dataset_provider() if dataset_provider else None
                    )
                )

        def on_default_simplification_change(event):
            fields["skeleton_mesh_simplification"].set_enabled(
                not bool(event.value)
            )
            # Re-selecting the default should immediately show the current
            # dataset-aware value instead of leaving the previous custom value.
            if bool(event.value):
                refresh_default_simplification()

        fields["use_default_simplification"].on_value_change(
            on_default_simplification_change
        )
        fields["skeleton_mesh_simplification"].set_enabled(False)

        for watcher in dataset_watchers or ():
            watcher.on_value_change(refresh_default_simplification)

    # Keep the returned object useful in tests and for callers that want to
    # style or position the expansion beside a checkbox.
    result = SkeletonVisualizationSettings(fields)
    result.panel = panel
    return result
