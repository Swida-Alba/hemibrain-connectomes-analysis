"""Visualization tabs - 3D Skeleton and Net-Viz (path network visualization).

Note: the interactive path-building Network tab lives in ui/tabs/network.py;
this module's pathway-graph tab is named Net-Viz (net_viz) to avoid any
name collision with it.
"""

import tempfile
from datetime import datetime
from pathlib import Path

from nicegui import ui

from ..config import (
    SKELETON_MODES,
    BRAIN_MESH_OPTIONS,
    NETWORK_LAYOUTS,
    SEARCH_COLUMNS,
    get_user_default,
    has_user_default,
)
from ..components.common import (
    dataset_selector, multi_select_input, number_input, select_input,
    checkbox_input, dir_input, read_upload_event, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..components.layer_tree_editor import layer_tree_editor
from ..components.palette_picker import (
    palette_picker,
    palette_editor,
    color_swatch_picker,
    assign_palette_colors,
    notify_empty_custom_palettes,
)
from ..components.edge_list_editor import edge_list_editor
from ..dataset_service import is_banc_dataset, is_flywire_dataset
from visualization_options import default_skeleton_tab_simplification
from ..roi_options import (
    load_roi_catalog,
    roi_options_from_catalog,
)


def _flatten_neuron_layers(neuron_layers) -> list:
    """Flatten the nested layer model into one neuron per entry for the
    per-neuron palette counts (single-neuron layers are plain values)."""
    return [
        n for layer in neuron_layers
        for n in (layer if isinstance(layer, list) else [layer])
    ]


def _palette_colors_for_count(palette, count: int) -> list:
    """Resolve a palette for a known render count.

    Continuous preset selections must span the selected range; discrete and
    custom palettes retain their displayed order/cycling behavior.
    """
    getter = getattr(palette, "get_colors_for_count", None)
    if callable(getter):
        return getter(count)
    return assign_palette_colors(palette.get_colors(), count)


COLOR_PRESETS = {
    "Cool": {"source": "#4A90E2", "intermediate": "#50E3C2", "target": "#B8E986", "link": "rgba(74,144,226,0.3)"},
    "Default": {"source": "#1f77b4", "intermediate": "rgba(44,160,44,1.0)", "target": "#d62728", "link": "rgba(100,100,100,0.6)"},
    "Warm": {"source": "#FF6B6B", "intermediate": "#FFA500", "target": "#FFD700", "link": "rgba(255,107,107,0.3)"},
    "Purple": {"source": "#9C27B0", "intermediate": "#BA68C8", "target": "#FF1493", "link": "rgba(156,39,176,0.3)"},
}

# Neuron color palettes (per layer), matched to utils.color_utils.generate_color_palette
# Path-network color schemes (source / intermediate / target / link)
PATH_SCHEMES = [
    (name, [scheme["source"], scheme["intermediate"], scheme["target"], scheme["link"]])
    for name, scheme in COLOR_PRESETS.items()
]

COMMON_ROIS = [
    "AL", "LH", "EB", "FB", "PB", "NO", "CA", "PED", "aL", "bL", "gL",
    "AB", "BU", "LAL", "AOTU", "AVLP", "PVLP", "PLP", "WED", "SLP",
    "SIP", "SMP", "CRE", "SCL", "ICL", "IB", "ATL", "VES", "EPA",
    "GOR", "SPS", "IPS", "SAD", "FLA", "CAN", "PRW", "GNG",
    "ME", "LO", "LOP", "AME",
]


def create_skeleton_tab():
    skeleton_runner = ScriptRunner()
    skeleton_output = OutputPanel("3D Skeleton Output")

    form_col, results_col = tool_page(
        "3D Skeleton",
        "Interactive neuron morphology, synapse, and brain-region rendering.",
        icon="view_in_ar",
        doc="skeleton.md",
    )

    with form_col:
        ui.label(
            "Configure and render neuron morphology independently from network drawings."
        ).classes("text-caption drocat-muted")

        with ui.card().classes("w-full drocat-card").props('id="card-skeleton-dataset"'):
            section_header("Dataset", "storage")
            dataset = dataset_selector(disable_banc=True)
            output_dir = dir_input(scope="visualization_skeleton")
            skeleton_dataset_warning = ui.label(
                "⚠️ BANC skeleton visualization is unavailable because FlyWire "
                "does not provide BANC skeletons. Select a non-BANC dataset."
            ).classes("text-caption text-amber-8").set_visibility(False)

        # ================= 3D Skeleton panel =================
        with ui.card().classes("w-full drocat-card").props('id="card-3d"'):
            section_header("3D Skeleton · plot3dSkeleton", "view_in_ar")
            ui.label(
                "Render neuron morphology, synapses, and independent brain-region meshes."
            ).classes("text-caption drocat-muted")
            section_header("Neuron Selection (3D)", "hub")
            layer_tree = layer_tree_editor(
                dataset_provider=lambda: dataset.value if dataset is not None else ""
            )
            with param_grid(3):
                filter_mode = select_input(
                    "Match by",
                    ["exact", "startswith", "contains", "endswith", "regex"],
                    "exact",
                    hint="The filter mode (exact / starts with / contains / ends with / "
                         "regex) applies to every neuron in the layer tree, matching "
                         "the pathfinding search backend.",
                )
                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, get_user_default("search_columns"),
                    hint="Which columns to search when resolving neuron names (same as "
                         "pathfinding). 'auto': all columns (bodyId -> type -> instance -> "
                         "flywireType/others). Use 'type'/'instance'/'bodyId' to restrict.",
                )
                hemisphere = select_input(
                    "Hemisphere", ["both", "left", "right"], "both",
                    hint="'both': plot all neurons. 'left'/'right': plot only that hemisphere. "
                         "Neurons WITHOUT an explicit hemisphere (no _L/_R instance suffix "
                         "or Soma side) are always included in every option.",
                )

            # ------------------------------------------------------------------
            # General Appearance (independent block)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-appearance"'):
                section_header("General Appearance", "palette")
                with param_grid(2):
                    skeleton_mode = select_input(
                        "Skeleton Mode", SKELETON_MODES, get_user_default("skeleton_mode"),
                        hint="'tube': 3D tube rendering (detailed). 'line': thin line (fast, for many neurons).",
                    )
                    legend_mode = select_input(
                        "Neuron Legend Mode", ["layer", "type", "single"], get_user_default("legend_mode"),
                        hint="'layer': one neuron legend entry per layer (or per custom group). "
                             "'type': per neuron type. 'single': every neuron. "
                             "ROI meshes always remain separate.",
                    )
                    bg_color = select_input(
                        "Background", ["white", "black"], get_user_default("background"),
                        hint="Background color for the 3D scene and exports. "
                             "The default neuron palette follows it: Category10 on "
                             "white, Set3 on black (until a palette is picked manually).",
                    )
                    brain_mesh = select_input(
                        "Brain Mesh", BRAIN_MESH_OPTIONS, get_user_default("brain_mesh"),
                        hint="'template': brain outline. 'whole': full brain surface. 'none': no mesh.",
                    )
                    vnc_mesh = checkbox_input(
                        "VNC Mesh", False,
                        hint="Show the ventral nerve cord mesh (male-cns / manc datasets).",
                    )

            # ------------------------------------------------------------------
            # Neuron Colors (independent block)
            # ------------------------------------------------------------------
            # Default neuron palette follows the background color: Category10
            # on white, Set3 on black.  Once the user picks a palette card
            # manually (on_change), the background stops switching it.
            palette_locked = {"locked": False}

            def _sync_palette_to_background():
                if palette_locked["locked"]:
                    return
                neuron_palette.set_palette(
                    "Set3" if bg_color.value == "black" else "Category10"
                )

            bg_color.on_value_change(lambda _e: _sync_palette_to_background())

            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-neuron-colors"'):
                section_header("Neuron Colors", "palette")
                neuron_palette = palette_editor(
                    "Neuron Colors",
                    value="Category10",
                    include_auto=False,
                    on_change=lambda: palette_locked.__setitem__("locked", True),
                )
                neuron_alpha = number_input(
                    "Neuron Opacity", 0.2, 0, 1, 0.1,
                    hint=(
                        "Global fallback opacity for skeletons (0=invisible, 1=solid). "
                        "A color with an explicit opacity channel overrides it; colors "
                        "without opacity inherit this value."
                    ),
                ).classes("w-48")
                ui.label(
                    "The default follows the background (Category10 on white, "
                    "Set3 on black) until a palette is picked manually. Custom colors "
                    "may include per-layer opacity; colors without an explicit "
                    "opacity channel use Neuron Opacity."
                ).classes("text-caption drocat-muted")

            # ------------------------------------------------------------------
            # Synapse Colors + synapse options (independent block)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-synapse-colors"'):
                section_header("Synapse Colors", "bubble_chart")
                synapse_palette = palette_editor(
                    "Synapse Colors",
                    value="Dark2",
                    include_auto=False,
                )
                ui.label(
                    "Colors are assigned per connection between consecutive layers "
                    "(one fewer than the number of neuron layers). Custom opacity "
                    "overrides Synapse Opacity per connection layer."
                ).classes("text-caption drocat-muted")
                ui.label("Synapse options").classes("drocat-mini-label")
                with param_grid(3):
                    skip_synapse = checkbox_input(
                        "Skip Synapses", False,
                        hint="Hide synapse markers for a cleaner view.",
                    )
                    min_synapse_num = number_input(
                        "Min Synapse Count", get_user_default("min_synapse_num"), 1, 100,
                        hint="Minimum synapses for a connection marker to be shown.",
                    )
                    synapse_size = select_input(
                        "Synapse Size", ["real", "1", "2", "3"], "real",
                        hint="'real': scale by synapse count. 1-3: fixed marker size.",
                    )
                    synapse_alpha = number_input(
                        "Synapse Opacity", 0.6, 0, 1, 0.1,
                        hint=(
                            "Global fallback opacity for synapse markers. A color with "
                            "an explicit opacity channel overrides it; colors without "
                            "opacity inherit it."
                        ),
                    )
                    synapse_mode = select_input(
                        "Synapse Mode", ["cone", "scatter"], "cone",
                        hint="'cone': directional cone markers. 'scatter': simple points.",
                    )

            # ------------------------------------------------------------------
            # Brain Region ROIs + ROI Colors (independent block)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-roi-colors"'):
                section_header("Brain Region ROIs (independent)", "view_in_ar")
                roi_select = multi_select_input(
                    "Mesh ROIs",
                    COMMON_ROIS,
                    default=[],
                    hint=(
                        "Select brain regions to show as meshes. No ROI meshes are "
                        "selected by default. Type any ROI name or regex (e.g. "
                        "ME.*, all, primary) and press Enter to add it."
                    ),
                ).props("outlined").props('new-value-mode="add-unique"')
                roi_detail_hint = ui.label(
                    "Primary ROI suggestions load the selected dataset's metadata."
                ).classes("text-caption drocat-muted")
                with ui.row().classes("w-full items-center gap-6"):
                    include_lr = checkbox_input(
                        "Include L/R variants",
                        False,
                        hint=(
                            "Keep explicit (L)/(R) entries in the primary list. "
                            "When off, bilateral pairs are shown once without "
                            "the suffix."
                        ),
                    )
                    include_subprimary = checkbox_input(
                        "Include sub-primary ROIs",
                        False,
                        hint=(
                            "Append non-primary ROI names from the local "
                            "available-ROI inventory to the primary list."
                        ),
                    )
                with param_grid(1):
                    mesh_alpha = number_input(
                        "ROI Mesh Opacity", 0.1, 0, 1, 0.05,
                        hint=(
                            "Global fallback opacity for ROI meshes. A color with an "
                            "explicit opacity channel overrides it; colors without "
                            "opacity inherit it."
                        ),
                    )
                roi_palette = palette_editor(
                    "ROI Colors",
                    value="Cool",
                    include_auto=True,
                )
                ui.label(
                    "Colors are assigned in the displayed order; every resolved ROI mesh "
                    "has its own legend entry. Use Custom colors for per-ROI opacity overrides."
                ).classes("text-caption drocat-muted")
                brain_mesh_picker = color_swatch_picker("Brain Mesh Color", value="auto")

            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                ui.label("Data & Rendering").classes("drocat-mini-label")
                with ui.row().classes("gap-4"):
                    cache_neurons = checkbox_input(
                        "Cache Neurons", get_user_default("cache_neurons"),
                        hint="Cache fetched skeletons locally for faster repeat renders.",
                    )
                    cache_default_state = {"user_changed": False, "updating": False}

                    def on_cache_neurons_change(_event):
                        if not cache_default_state["updating"]:
                            cache_default_state["user_changed"] = True

                    cache_neurons.on_value_change(on_cache_neurons_change)
                    cache_synapses = checkbox_input(
                        "Cache Synapses", get_user_default("cache_synapses"),
                        hint="Cache fetched synapse data locally.",
                    )
                    smooth_skeleton = checkbox_input(
                        "Smooth Skeleton", get_user_default("smooth_skeleton"),
                        hint="Apply mesh smoothing to neuron skeletons.",
                    )
                    show_soma = checkbox_input(
                        "Show Soma", get_user_default("show_soma"),
                        hint="Render the soma sphere for neurons that have one.",
                    )
                    show_connectors = checkbox_input(
                        "Show Connectors", get_user_default("show_connectors"),
                        hint="Show synaptic connector markers.",
                    )
                with ui.row().classes("gap-4"):
                    simplification_method = select_input(
                        "Simplification Method",
                        ["fast", "fine", "artistic"],
                        get_user_default("simplification_method"),
                        hint=(
                            "NeuPrint tube rendering: 'fast' (default) uses "
                            "direct simp90 simplification plus the FAFB fast "
                            "node-reduction stage; 'fine' smooths/resamples "
                            "with the accelerated FAFB radius profile; "
                            "'artistic' uses vertex-cluster mesh decimation. "
                            "All methods use batched parallel online "
                            "fetching and are available for NeuPrint and "
                            "FlyWire/FAFB tube renders; line mode bypasses "
                            "the method."
                        ),
                    )
                    default_simplification = checkbox_input(
                        "Use Default Mesh Simplification", True,
                        hint="Use the method default: fast removes 0.90 of faces; "
                             "fine/artistic remove 0.95 for NeuPrint and "
                             "FlyWire/FAFB. Uncheck to set the value below.",
                    )
                    mesh_simplification = number_input(
                        "Mesh Simplification (faces removed)",
                        default_skeleton_tab_simplification(
                            dataset.value, simplification_method.value,
                        ),
                        0.0, 0.99, 0.05,
                        hint="Fraction of tube-mesh faces REMOVED for rendering: "
                             "0.95 = keep 5%. Higher = faster/coarser, lower = "
                             "more detailed but slower.",
                    )
                    mesh_simplification.set_enabled(False)

                ui.label("Export").classes("drocat-mini-label")
                with param_grid(3):
                    export_method = select_input(
                        "Export Method", ["webdriver", "kaleido"], "webdriver",
                        hint="'webdriver': fast, needs Chrome 109+. 'kaleido': slower but stable fallback.",
                    )
                    export_scale = number_input(
                        "Export Scale", 3, 1, 5,
                        hint="Resolution multiplier for PNG exports (higher = sharper).",
                    )
                with ui.row().classes("gap-4"):
                    show_fig = checkbox_input(
                        "Show Figure", get_user_default("show_fig_skeleton"),
                        hint="Open the 3D HTML visualization after rendering.",
                    )
                    export_views = checkbox_input(
                        "Export Views", get_user_default("export_views"),
                        hint="Export PNG screenshots from 6 angles.",
                    )

            # ------------------------------------------------------------------
            # Export Video / GIF + Individual Profiles (independent block,
            # outside advanced settings)
            # ------------------------------------------------------------------
            with ui.card().classes("w-full drocat-card").props('id="card-skeleton-export-video"'):
                section_header("Export Video / GIF", "videocam")
                with ui.row().classes("gap-4"):
                    export_video = checkbox_input(
                        "Export Video", False,
                        hint="Render a rotating video of the 3D scene (needs Chrome/WebDriver).",
                    )
                    rotate = select_input(
                        "Rotate", ["horizontal", "vertical"], "horizontal",
                    )
                    export_gif = checkbox_input(
                        "Also Export GIF", True,
                        hint="Convert the video to a small GIF as well.",
                    )
                with param_grid(3):
                    fps = number_input("FPS", 30, 5, 60, 5)
                    degree_per_frame = number_input("Degrees / Frame", 1.0, 0.1, 5.0, 0.1)
                    gif_scale = number_input("GIF Scale", 0.2, 0.05, 1.0, 0.05)

                ui.separator().classes("my-2")
                section_header("Individual Profiles (PDF / PPTX)", "photo_library")
                with ui.row().classes("gap-4"):
                    export_individual_profiles = checkbox_input(
                        "Export Individual Profiles", False,
                        hint="After rendering, generate a PDF/PPTX with per-neuron profile plots.",
                    )
                    summary_format = multi_select_input(
                        "Summary Format", ["pdf", "pptx"], ["pdf"],
                        hint="Output formats for the individual-profile summary.",
                    )
                with param_grid(3):
                    profile_cols = number_input("Images Per Page (cols)", 3, 1, 6)
                    profile_rows = number_input("Images Per Page (rows)", 2, 1, 6)
                    profile_views = multi_select_input(
                        "Profile Views", ["front", "side", "top", "back", "bottom"], ["front"],
                        hint="Camera views included in each individual profile.",
                    )
                ui.label(
                    "Each individual profile follows the Neuron Legend Mode: "
                    "'single' = one profile per neuron, 'type' = one profile per type "
                    "(all layers combined), 'layer' = one profile per layer / custom group."
                ).classes("text-caption drocat-muted")

        def _sync_simplification_controls():
            is_line = skeleton_mode.value == "line"
            pipeline = str(simplification_method.value or "fine").strip().lower()
            if default_simplification.value:
                mesh_simplification.set_value(
                    default_skeleton_tab_simplification(
                        dataset.value, pipeline,
                    )
                )
            mesh_simplification.set_enabled(
                not is_line and not default_simplification.value
            )
            default_simplification.set_enabled(not is_line)
            simplification_method.set_enabled(
                not is_line
            )
            if (
                not cache_default_state["user_changed"]
                and not has_user_default("cache_neurons")
            ):
                # FAFB's method selector starts at ``fast`` and remains
                # selectable; its fast/fine default follows the selected
                # method while cache eligibility is handled independently.
                default_cache = (
                    True if is_flywire_dataset(str(dataset.value or ""))
                    else pipeline not in {"fast", "artistic"}
                )
                if cache_neurons.value != default_cache:
                    cache_default_state["updating"] = True
                    try:
                        cache_neurons.set_value(default_cache)
                    finally:
                        cache_default_state["updating"] = False

        default_simplification.on_value_change(
            lambda _e: _sync_simplification_controls()
        )
        skeleton_mode.on_value_change(lambda _e: _sync_simplification_controls())
        simplification_method.on_value_change(
            lambda _e: _sync_simplification_controls()
        )
        dataset.on_value_change(lambda _e: _sync_simplification_controls())
        _sync_simplification_controls()

        def _sync_roi_options():
            catalog = load_roi_catalog(dataset.value)
            options = roi_options_from_catalog(
                catalog,
                include_lr=bool(include_lr.value),
                include_subprimary=bool(include_subprimary.value),
                fallback=COMMON_ROIS,
            )
            primary_count = len(catalog["primary"])
            if catalog["primary"]:
                suffix = ""
                if include_subprimary.value:
                    suffix = " plus sub-primary entries"
                roi_detail_hint.set_text(
                    f"Primary list: {primary_count} metadata ROIs{suffix} "
                    f"for {dataset.value}."
                )
            else:
                roi_detail_hint.set_text(
                    "No local metadata primary list was found; using the "
                    "common fallback list."
                )
            # QSelect only renders chips whose values exist in its options,
            # so keep the currently selected ROIs in the new option list.
            for value in roi_select.value or []:
                if value not in options:
                    options.append(value)
            roi_select.set_options(options)

        def _sync_skeleton_dataset_warning():
            skeleton_dataset_warning.set_visibility(
                is_banc_dataset(dataset.value)
            )

        include_lr.on_value_change(lambda _e: _sync_roi_options())
        include_subprimary.on_value_change(lambda _e: _sync_roi_options())
        dataset.on_value_change(lambda _e: _sync_roi_options())
        dataset.on_value_change(lambda _e: _sync_skeleton_dataset_warning())
        _sync_roi_options()
        _sync_skeleton_dataset_warning()

    with results_col:
        skeleton_output.create(run_label="Generate 3D Skeleton", run_icon="view_in_ar")

    async def run_panel(
        output_panel,
        runner,
        tool_name,
        constructor_params,
        output_dir_for_scan,
        method_params=None,
    ):
        output_panel.clear()
        output_panel.set_running(True)
        result = await output_panel.run(
            runner,
            tool_name,
            constructor_params,
            "plot",
            method_params=method_params,
            output_dir=output_dir_for_scan,
        )
        output_panel.set_running(False)
        output_panel.set_status(
            "Completed" if result["returncode"] == 0 else "Failed",
            "green" if result["returncode"] == 0 else "red",
        )
        output_panel.show_files(
            result["files"],
            result.get("output_folder") or output_dir_for_scan,
        )
        return result

    async def run_skeleton():
        if is_banc_dataset(dataset.value):
            _sync_skeleton_dataset_warning()
            ui.notify(
                "BANC skeleton visualization is unavailable; select a non-BANC dataset.",
                type="warning",
            )
            return
        # The layer tree maps 1:1 to the backend's nested-list model:
        # neuron_layers[i] = layer i's neurons, custom_layer_names partial.
        neuron_layers = layer_tree.get_neuron_layers()
        if not neuron_layers:
            ui.notify("Add at least one neuron to a layer", type="warning")
            return
        # Raw chips (pre-pattern) for the query history: the filter-mode
        # conversion below rewrites them into regex patterns, which are not
        # useful history entries.
        raw_neurons = _flatten_neuron_layers(neuron_layers)
        # Same search semantics as pathfinding: the filter mode converts
        # every neuron into the regex pattern resolved by statvis.getNeurons.
        mode = filter_mode.value
        if mode and mode != "exact":
            converted = []
            for layer in neuron_layers:
                as_list = layer if isinstance(layer, list) else [layer]
                filtered = apply_filter_mode(as_list, mode)
                converted.append(
                    filtered if isinstance(layer, list) else filtered[0]
                )
            neuron_layers = converted
        custom_names = layer_tree.get_custom_layer_names()
        rois = roi_select.value or []

        # Flatten the (filter-converted) layers for the per-neuron palette:
        # one neuron color per neuron; synapse colors span the gaps between
        # consecutive neurons (same counts as the old flat chip list).
        neurons = _flatten_neuron_layers(neuron_layers)

        # Assign the exact displayed palette order (including custom reordering).
        neuron_colors = _palette_colors_for_count(neuron_palette, len(neurons))
        # One color per connection between consecutive layers (n_layers - 1).
        synapse_colors = _palette_colors_for_count(
            synapse_palette, max(0, len(neurons) - 1)
        )

        # Auto is a one-color gray palette; custom mode must not be gated by
        # the last selected preset name.
        mesh_color = _palette_colors_for_count(
            roi_palette, len(rois)
        ) if rois else (100, 100, 100)

        # An empty custom palette renders with the backend's default palette
        # (which logs a warning internally); surface it in the UI too so the
        # substitution is not silent. The ROI palette is only consulted when
        # at least one ROI mesh is selected.
        palettes = [
            (neuron_palette, "Neuron Colors"),
            (synapse_palette, "Synapse Colors"),
        ]
        if rois:
            palettes.append((roi_palette, "ROI Colors"))
        notify_empty_custom_palettes(*palettes)

        custom_names = layer_tree.get_custom_layer_names()

        constructor_params = {
            "dataset": dataset.value,
            "neuron_layers": neuron_layers,
            "search_columns": search_columns.value,
            "hemisphere": hemisphere.value,
            "custom_layer_names": custom_names,
            "output_dir": output_dir.value,
            "output_format": get_user_default("output_format"),
            "skeleton_mode": skeleton_mode.value,
            "brain_mesh": brain_mesh.value,
            "vnc_mesh": vnc_mesh.value,
            "legend_mode": legend_mode.value,
            "neuron_alpha": float(neuron_alpha.value),
            "neuron_colors": neuron_colors,
            "synapse_colors": synapse_colors,
            "background_color": bg_color.value,
            "skip_synapse": skip_synapse.value,
            "min_synapse_num": int(min_synapse_num.value),
            "synapse_size": synapse_size.value,
            "synapse_alpha": float(synapse_alpha.value),
            "synapse_mode": synapse_mode.value,
            "mesh_roi": rois,
            "mesh_color": mesh_color,
            "mesh_alpha": float(mesh_alpha.value),
            "cache_neurons": cache_neurons.value,
            "cache_synapses": cache_synapses.value,
            "smooth_skeleton": smooth_skeleton.value,
            "show_soma": show_soma.value,
            "show_connectors": show_connectors.value,
            "export_method": export_method.value,
            "export_scale": int(export_scale.value),
            "export_views": export_views.value,
            "show_fig": show_fig.value,
            "brain_mesh_color": brain_mesh_picker.get_value(),
            "neuprint_skeleton_pipeline": simplification_method.value,
            "skeleton_mesh_simplification": (
                default_skeleton_tab_simplification(
                    dataset.value, simplification_method.value,
                ) if default_simplification.value
                else float(mesh_simplification.value)
            ),
        }
        method_params = {
            "export_individual_profiles": export_individual_profiles.value,
            "pdf_images_per_page": (
                int(profile_cols.value),
                int(profile_rows.value),
            ),
            "views": profile_views.value or ["front"],
            "summary_format": summary_format.value or ["pdf"],
            "export_video": export_video.value,
            "fps": int(fps.value),
            "degree_per_frame": float(degree_per_frame.value),
            "rotate": rotate.value,
            "export_gif": export_gif.value,
            "gif_scale": float(gif_scale.value),
        }
        result = await run_panel(
            skeleton_output,
            skeleton_runner,
            "plot3d_skeleton",
            constructor_params,
            output_dir.value,
            method_params,
        )
        # A completed render means the layer neurons resolved in the dataset;
        # keep the raw chips in the query history like the pathfinding tabs.
        if result["returncode"] == 0:
            from ..history_store import record as _record_history
            _record_history(
                [str(n) for n in dict.fromkeys(raw_neurons)],
                datasets=[dataset.value] if dataset.value else [],
            )

    skeleton_output.run_button.on_click(run_skeleton)
    skeleton_output.cancel_button.on_click(skeleton_runner.cancel)


def create_net_viz_tab():
    """Create the standalone PlotPath pathway-graph tab (Net-Viz)."""
    net_viz_runner = ScriptRunner()
    net_viz_output = OutputPanel("Net-Viz Output")
    path_file_path = {"path": None}

    form_col, results_col = tool_page(
        "Net-Viz",
        "Interactive pathway graphs and editable empty drawing canvases.",
        icon="account_tree",
        doc="network.md",
    )

    with form_col:
        ui.label(
            "Load a path result, build a custom edge list, or create an empty "
            "HTML canvas for direct interactive drawing."
        ).classes("text-caption drocat-muted")

        with ui.card().classes("w-full drocat-card").props('id="card-net-viz-output"'):
            section_header("Output Directory", "folder")
            path_output_dir = dir_input(
                label="Path Output Directory",
                scope="visualization_path",
            )

        with ui.card().classes("w-full drocat-card").props('id="card-net-viz-source"'):
            section_header("Net-Viz Source", "source")
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                with ui.element("div").classes("grow min-w-[240px]"):
                    net_viz_source = select_input(
                        "Canvas Source",
                        ["Path file", "Edge list editor"],
                        "Path file",
                        hint=(
                            "Path file: load Complete Paths output. Edge list editor: "
                            "build or edit an edge list with auto-save."
                        ),
                    )
                empty_canvas_button = ui.button(
                    "Create Empty Canvas",
                    icon="open_in_new",
                ).props("color=secondary outline").classes("drocat-empty-canvas-btn")
            ui.label(
                "Path file mode accepts CSV or Excel with path_type / path_bodyId sheets."
            ).classes("text-caption drocat-muted")

            async def handle_path_upload(e):
                temporary_path = None
                previous_path = path_file_path.get("path")
                try:
                    filename, data = await read_upload_event(e)
                    suffix = Path(filename).suffix.lower()
                    if suffix not in {".csv", ".xlsx", ".xls"}:
                        raise ValueError("Use a CSV, XLSX, or XLS path file")
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, prefix="drocat_paths_", delete=False
                    ) as temporary:
                        temporary.write(data)
                        temporary_path = temporary.name

                    path_file_path["path"] = temporary_path
                    if previous_path and previous_path != temporary_path:
                        Path(previous_path).unlink(missing_ok=True)
                    path_upload_label.text = (
                        f"Loaded: {filename} ({len(data) / 1024:.1f} KB)"
                    )
                    path_upload_label.classes(replace="text-caption drocat-ok")
                except Exception as ex:
                    if temporary_path:
                        Path(temporary_path).unlink(missing_ok=True)
                    if previous_path:
                        Path(previous_path).unlink(missing_ok=True)
                    path_file_path["path"] = None
                    path_upload_label.text = f"Error: {ex}"
                    path_upload_label.classes(replace="text-caption drocat-err")
                path_upload_menu.close()

            with ui.column().classes("w-full gap-2").props(
                'id="net-viz-path-input"'
            ) as path_input_panel:
                with ui.row().classes("w-full items-center gap-2"):
                    with ui.button(icon="upload_file").props("flat dense round").classes(
                        "drocat-upload-trigger"
                    ).tooltip("Upload a *_allpaths_info file (CSV/Excel)"):
                        with ui.menu() as path_upload_menu:
                            ui.label("Load path data from Complete Paths").classes(
                                "text-caption drocat-muted px-3 pt-2"
                            )
                            ui.label(
                                "CSV / XLSX / XLS with path_type or path_bodyId sheets"
                            ).classes("text-caption drocat-muted px-3 pb-1")
                            ui.upload(
                                label="Choose paths file",
                                on_upload=handle_path_upload,
                                auto_upload=True,
                            ).props('accept=".csv,.xlsx,.xls" flat dense').classes("w-72")
                            ui.link(
                                "File format instructions",
                                "docs/ui_guides/network.html#input-files",
                            ).classes("drocat-doc-link px-3 pb-2")
                    path_upload_label = ui.label("No path file selected.").classes(
                        "text-caption drocat-muted drocat-truncate"
                    )

            def update_net_viz_source():
                source = net_viz_source.value
                path_input_panel.set_visibility(source == "Path file")
                if source != "Path file":
                    path_upload_menu.close()

        editor_panel_ref = {"panel": None}

        def expand_editor_for_source():
            panel = editor_panel_ref.get("panel")
            if panel is not None and not panel.value:
                panel.set_value(True)

        def collapse_editor_for_source():
            panel = editor_panel_ref.get("panel")
            if panel is not None and panel.value:
                panel.set_value(False)

        def update_net_viz_source_with_editor():
            update_net_viz_source()
            if net_viz_source.value == "Edge list editor":
                expand_editor_for_source()
            else:
                collapse_editor_for_source()

        # Keep the source selector and the editor expansion synchronized. The
        # expansion callback handles the important reverse direction: opening
        # the editor immediately selects it as the run source.
        net_viz_source.on_value_change(
            lambda _event: update_net_viz_source_with_editor()
        )

        # Edge-list CSV exports are browser downloads. The configured output
        # directory is reserved for artifacts generated by the Net-Viz run.
        editor = edge_list_editor(
            on_expand=lambda: net_viz_source.set_value("Edge list editor"),
        )
        editor_panel_ref["panel"] = editor.expansion
        update_net_viz_source()

        with ui.card().classes("w-full drocat-card").props('id="card-net-viz-rendering"'):
            section_header("Rendering Options", "palette")
            path_layout = select_input(
                "Network Layout",
                NETWORK_LAYOUTS + ["hierarchical"],
                "hierarchical",
                hint="Layout algorithm for the HTML network visualization.",
            )
            color_preset = palette_picker(
                "Color Scheme",
                value="Cool",
                catalog=PATH_SCHEMES,
            )
            with ui.row().classes("gap-4"):
                show_path_fig = checkbox_input(
                    "Open in Browser",
                    True,
                    hint="Open the interactive HTML network after rendering.",
                )
            ui.label(
                "Path mode exports HTML and connection tables; empty mode exports an HTML canvas."
            ).classes("text-caption drocat-muted")

    with results_col:
        net_viz_output.create(run_label="Generate Network", run_icon="account_tree")

    async def run_panel(constructor_params):
        net_viz_output.clear()
        net_viz_output.set_running(True)
        result = await net_viz_output.run(
            net_viz_runner,
            "plot_path",
            constructor_params,
            "plot",
            output_dir=path_file_path.get("output_folder") or path_output_dir.value,
        )
        net_viz_output.set_running(False)
        net_viz_output.set_status(
            "Completed" if result["returncode"] == 0 else "Failed",
            "green" if result["returncode"] == 0 else "red",
        )
        net_viz_output.show_files(
            result["files"],
            result.get("output_folder") or path_file_path.get("output_folder") or path_output_dir.value,
        )

    def make_plotpath_folder(empty_canvas=False):
        """Create the per-run output subfolder (plot-network_{name}_{timestamp}).

        Every run gets its own timestamped folder inside the user-chosen
        output directory, matching the naming of the other tools.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if empty_canvas:
            name = "empty_network"
        else:
            name = Path(path_file_path["path"]).stem or "paths"
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            if len(name) > 60:
                name = name[:60]
        run_folder = Path(path_output_dir.value) / f"plot-network_{name}_{timestamp}"
        run_folder.mkdir(parents=True, exist_ok=True)
        path_file_path["output_folder"] = str(run_folder)
        return str(run_folder)

    async def execute_net_viz(empty_canvas=False):
        editor_mode = (not empty_canvas) and net_viz_source.value == "Edge list editor"
        transient_editor_path = None
        if editor_mode:
            # Flush the auto-save when named; otherwise use a temporary CSV.
            # A draft name is optional for running the edited edge list.
            csv_path = editor.runnable_path_file()
            if not csv_path:
                ui.notify(
                    "Add at least one complete edge (source, target, weight) first",
                    type="warning",
                )
                return
            path_file_path["path"] = csv_path
            transient_editor_path = csv_path if editor.transient_csv_path == csv_path else None
        elif not empty_canvas and not path_file_path["path"]:
            ui.notify(
                "Please upload a path file first (Complete Paths output)",
                type="warning",
            )
            return

        try:
            colors = COLOR_PRESETS.get(color_preset.get_value(), COLOR_PRESETS["Cool"])
            constructor_params = {
                "path_file": None if empty_canvas else path_file_path["path"],
                "output_folder": make_plotpath_folder(empty_canvas),
                "source_color": colors["source"],
                "intermediate_color": colors["intermediate"],
                "target_color": colors["target"],
                "link_color": colors["link"],
                "network_layout": path_layout.value,
                "showfig": True if empty_canvas else show_path_fig.value,
                "generate_empty_network": empty_canvas,
            }
            await run_panel(constructor_params)
        finally:
            if transient_editor_path:
                editor.cleanup_transient_csv()
                if path_file_path.get("path") == transient_editor_path:
                    path_file_path["path"] = None

    async def run_net_viz():
        await execute_net_viz(empty_canvas=False)


    async def create_empty_canvas():
        await execute_net_viz(empty_canvas=True)

    net_viz_output.run_button.on_click(run_net_viz)
    net_viz_output.cancel_button.on_click(net_viz_runner.cancel)
    empty_canvas_button.on_click(create_empty_canvas)


def create_visualization_tab():
    """Backward-compatible combined view for callers outside the main app."""
    create_skeleton_tab()
    create_net_viz_tab()
