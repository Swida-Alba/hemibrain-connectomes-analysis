"""Visualization Tab - 3D skeleton and path network visualization."""

import tempfile
from datetime import datetime
from pathlib import Path

from nicegui import ui

from ..config import SKELETON_MODES, BRAIN_MESH_OPTIONS, NETWORK_LAYOUTS, SEARCH_COLUMNS
from ..components.common import (
    dataset_selector, neuron_list_input, multi_select_input, number_input, select_input,
    checkbox_input, dir_input, read_upload_event, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..components.palette_picker import (
    palette_picker,
    palette_editor,
    color_swatch_picker,
    assign_palette_colors,
)


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
        doc="visualization.md",
    )

    with form_col:
        ui.label(
            "Configure and render neuron morphology independently from network drawings."
        ).classes("text-caption drocat-muted")

        # ================= 3D Skeleton panel =================
        with ui.card().classes("w-full drocat-card").props('id="card-3d"'):
            section_header("3D Skeleton · plot3dSkeleton", "view_in_ar")
            ui.label(
                "Render neuron morphology, synapses, and independent brain-region meshes."
            ).classes("text-caption drocat-muted")
            section_header("Neuron Selection (3D)", "hub")
            neuron_chips = neuron_list_input(
                label="Neurons / Layers",
                show_upload=False,
                hint="Type a neuron name and press Enter (or leave the field) to add a chip. "
                     "Each chip = one neuron/layer; use 'A -> B -> C' for connected paths. "
                     "The filter mode (exact / starts with / contains / ends with / regex) "
                     "applies to every chip, matching the pathfinding search backend.",
            )
            custom_layer_names = neuron_list_input(
                label="Custom Layer Names (optional)",
                show_filter=False,
                show_upload=False,
                hint="Display names for each neuron layer, in the same order as the chips.",
            )
            with param_grid(2):
                dataset = dataset_selector()
                output_dir = dir_input()
            search_columns = select_input(
                "Search Columns", SEARCH_COLUMNS, "auto",
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
                        "Skeleton Mode", SKELETON_MODES, "tube",
                        hint="'tube': 3D tube rendering (detailed). 'line': thin line (fast, for many neurons).",
                    )
                    legend_mode = select_input(
                        "Neuron Legend Mode", ["layer", "type", "single"], "layer",
                        hint="'layer': one neuron legend entry per layer. 'type': per neuron type. "
                             "'single': every neuron. ROI meshes always remain separate.",
                    )
                    neuron_alpha = number_input(
                        "Neuron Opacity", 0.2, 0, 1, 0.1,
                        hint="Transparency of neuron tubes (0=invisible, 1=solid).",
                    )
                    bg_color = select_input(
                        "Background", ["white", "black"], "white",
                        hint="Background color for the 3D scene and exports. "
                             "The default neuron palette follows it: Category10 on "
                             "white, Set3 on black (until a palette is picked manually).",
                    )
                    brain_mesh = select_input(
                        "Brain Mesh", BRAIN_MESH_OPTIONS, "template",
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
                ui.label(
                    "The default follows the background (Category10 on white, "
                    "Set3 on black) until a palette is picked manually."
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
                    "(one fewer than the number of neuron layers)."
                ).classes("text-caption drocat-muted")
                ui.label("Synapse options").classes("drocat-mini-label")
                with param_grid(3):
                    skip_synapse = checkbox_input(
                        "Skip Synapses", True,
                        hint="Hide synapse markers for a cleaner view.",
                    )
                    min_synapse_num = number_input(
                        "Min Synapse Count", 3, 1, 100,
                        hint="Minimum synapses for a connection marker to be shown.",
                    )
                    synapse_size = select_input(
                        "Synapse Size", ["real", "1", "2", "3"], "real",
                        hint="'real': scale by synapse count. 1-3: fixed marker size.",
                    )
                    synapse_alpha = number_input(
                        "Synapse Opacity", 0.6, 0, 1, 0.1,
                        hint="Transparency of synapse markers.",
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
                    default=["EB", "LH", "AL"],
                    hint="Select brain regions to show as meshes. Type any ROI name or regex "
                         "(e.g. ME.*, all, primary) and press Enter to add it.",
                )
                roi_select.props('new-value-mode="add-unique"')
                roi_palette = palette_editor(
                    "ROI Colors",
                    value="Cool",
                    include_auto=True,
                )
                ui.label(
                    "Colors are assigned in the displayed order; every resolved ROI mesh "
                    "has its own legend entry."
                ).classes("text-caption drocat-muted")
                with param_grid(3):
                    mesh_alpha = number_input(
                        "ROI Mesh Opacity", 0.1, 0, 1, 0.05,
                        hint="Transparency applied to all ROI meshes (per-ROI alpha can be "
                             "embedded in custom colors instead).",
                    )

            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                ui.label("Data & Rendering").classes("drocat-mini-label")
                with ui.row().classes("gap-4"):
                    cache_neurons = checkbox_input(
                        "Cache Neurons", True,
                        hint="Cache fetched skeletons locally for faster repeat renders.",
                    )
                    cache_synapses = checkbox_input(
                        "Cache Synapses", True,
                        hint="Cache fetched synapse data locally.",
                    )
                    smooth_skeleton = checkbox_input(
                        "Smooth Skeleton", False,
                        hint="Apply mesh smoothing to neuron skeletons.",
                    )
                    show_soma = checkbox_input(
                        "Show Soma", True,
                        hint="Render the soma sphere for neurons that have one.",
                    )
                    show_connectors = checkbox_input(
                        "Show Connectors", False,
                        hint="Show synaptic connector markers.",
                    )

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
                brain_mesh_picker = color_swatch_picker("Brain Mesh Color", value="auto")
                with ui.row().classes("gap-4"):
                    show_fig = checkbox_input(
                        "Show Figure", True,
                        hint="Open the 3D HTML visualization after rendering.",
                    )
                    export_views = checkbox_input(
                        "Export Views", True,
                        hint="Export PNG screenshots from 6 angles.",
                    )

                ui.label("Individual Profiles (PDF / PPTX)").classes("drocat-mini-label")
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

                ui.label("Rotating Video / GIF").classes("drocat-mini-label")
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

    async def run_skeleton():
        mode, neurons = neuron_chips.get_value()
        if not neurons:
            ui.notify("Please add at least one neuron", type="warning")
            return
        # Same search semantics as pathfinding: the filter mode (exact /
        # starts with / contains / ends with / regex) converts every chip
        # into the regex pattern resolved by statvis.getNeurons.
        neurons = apply_filter_mode(neurons, mode)
        rois = roi_select.value or []

        # Assign the exact displayed palette order (including custom reordering).
        neuron_colors = assign_palette_colors(
            neuron_palette.get_colors(), len(neurons)
        )
        # One color per connection between consecutive layers (n_layers - 1).
        synapse_colors = assign_palette_colors(
            synapse_palette.get_colors(), max(0, len(neurons) - 1)
        )

        # Auto is a one-color gray palette; custom mode must not be gated by
        # the last selected preset name.
        mesh_color = assign_palette_colors(
            roi_palette.get_colors(), len(rois)
        ) if rois else (100, 100, 100)

        custom_names = [str(n) for n in custom_layer_names.get_value()[1]]

        constructor_params = {
            "dataset": dataset.value,
            "neuron_layers": neurons,
            "search_columns": search_columns.value,
            "hemisphere": hemisphere.value,
            "custom_layer_names": custom_names,
            "output_dir": output_dir.value,
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
        await run_panel(
            skeleton_output,
            skeleton_runner,
            "plot3d_skeleton",
            constructor_params,
            output_dir.value,
            method_params,
        )

    skeleton_output.run_button.on_click(run_skeleton)
    skeleton_output.cancel_button.on_click(skeleton_runner.cancel)


def create_network_tab():
    """Create the standalone PlotPath network visualization tab."""
    network_runner = ScriptRunner()
    network_output = OutputPanel("Network Output")
    path_file_path = {"path": None}

    form_col, results_col = tool_page(
        "Network Visualization",
        "Interactive pathway graphs and editable empty drawing canvases.",
        icon="account_tree",
        doc="visualization.md",
    )

    with form_col:
        ui.label(
            "Load a path result or create an empty HTML canvas for direct interactive drawing."
        ).classes("text-caption drocat-muted")

        with ui.card().classes("w-full drocat-card").props('id="card-network"'):
            section_header("Network Visualization · PlotPath", "account_tree")
            network_source = select_input(
                "Canvas Source",
                ["Path file", "Empty drawing canvas"],
                "Path file",
                hint=(
                    "Path file: load Find All Paths output. Empty drawing canvas: "
                    "add nodes and edges in the generated HTML."
                ),
            )
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

            with ui.column().classes("w-full gap-2") as path_input_panel:
                with ui.row().classes("w-full items-center gap-2"):
                    with ui.button(icon="upload_file").props("flat dense round").classes(
                        "drocat-upload-trigger"
                    ).tooltip("Upload a *_allpaths_info file (CSV/Excel)"):
                        with ui.menu() as path_upload_menu:
                            ui.label("Load path data from Find All Paths").classes(
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
                                "docs/ui_guides/input_formats.html",
                            ).classes("drocat-doc-link px-3 pb-2")
                    path_upload_label = ui.label("No path file selected.").classes(
                        "text-caption drocat-muted drocat-truncate"
                    )

            empty_canvas_hint = ui.label(
                "Empty canvas mode creates an editable Cytoscape HTML. Open it, enable "
                "Edit Mode, then add nodes and connect them interactively."
            ).classes("text-caption drocat-muted")

            with param_grid(2):
                path_output_dir = dir_input(label="Path Output Directory")
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

            empty_canvas_button = ui.button(
                "Create Empty Canvas",
                icon="open_in_new",
            ).props("color=secondary outline").classes("w-full")

            def update_network_source():
                is_empty = network_source.value == "Empty drawing canvas"
                path_input_panel.set_visibility(not is_empty)
                empty_canvas_hint.set_visibility(is_empty)
                if is_empty:
                    path_upload_menu.close()

            network_source.on_value_change(lambda _event: update_network_source())
            update_network_source()

    with results_col:
        network_output.create(run_label="Generate Network", run_icon="account_tree")

    async def run_panel(constructor_params):
        network_output.clear()
        network_output.set_running(True)
        result = await network_output.run(
            network_runner,
            "plot_path",
            constructor_params,
            "plot",
            output_dir=path_file_path.get("output_folder") or path_output_dir.value,
        )
        network_output.set_running(False)
        network_output.set_status(
            "Completed" if result["returncode"] == 0 else "Failed",
            "green" if result["returncode"] == 0 else "red",
        )
        network_output.show_files(
            result["files"],
            result.get("output_folder") or path_file_path.get("output_folder") or path_output_dir.value,
        )

    def make_plotpath_folder(empty_canvas=False):
        """Create the per-run output subfolder (plotpath_{name}_{timestamp}).

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
        run_folder = Path(path_output_dir.value) / f"plotpath_{name}_{timestamp}"
        run_folder.mkdir(parents=True, exist_ok=True)
        path_file_path["output_folder"] = str(run_folder)
        return str(run_folder)

    async def execute_network(empty_canvas=False):
        if not empty_canvas and not path_file_path["path"]:
            ui.notify(
                "Please upload a path file first (Find All Paths output)",
                type="warning",
            )
            return

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

    async def run_network():
        await execute_network(network_source.value == "Empty drawing canvas")

    async def create_empty_canvas():
        await execute_network(empty_canvas=True)

    network_output.run_button.on_click(run_network)
    network_output.cancel_button.on_click(network_runner.cancel)
    empty_canvas_button.on_click(create_empty_canvas)


def create_visualization_tab():
    """Backward-compatible combined view for callers outside the main app."""
    create_skeleton_tab()
    create_network_tab()
