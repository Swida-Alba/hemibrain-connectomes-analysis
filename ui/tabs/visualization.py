"""Visualization Tab - 3D skeleton and path network visualization."""

import tempfile
from pathlib import Path

from nicegui import ui

from ..config import SKELETON_MODES, BRAIN_MESH_OPTIONS, NETWORK_LAYOUTS
from ..components.common import (
    dataset_selector, chip_list_input, multi_select_input, number_input, select_input,
    checkbox_input, dir_input, read_upload_event, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..components.palette_picker import (
    palette_picker,
    palette_editor,
    color_swatch_picker,
    sample_palette,
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


def create_visualization_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Visualization Output")
    path_file_path = {"path": None}

    form_col, results_col = tool_page(
        "Visualization",
        "3D neuron skeleton rendering and path network plotting.",
        icon="view_in_ar",
        doc="visualization.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Tool Selection", "brush")
            tool_select = ui.select(
                options=["3D Skeleton (plot3dSkeleton)", "Path Network (PlotPath)"],
                value="3D Skeleton (plot3dSkeleton)",
                label="Tool",
            ).classes("w-full drocat-select").tooltip(
                "3D Skeleton: render neurons in 3D with brain mesh. "
                "Path Network: plot a connection graph from a FindAllPath result file."
            )

        # ================= 3D Skeleton options =================
        with ui.card().classes("w-full drocat-card").props('id="card-3d"'):
            section_header("Neuron Selection (3D)", "hub")
            neuron_chips = chip_list_input(
                label="Neurons / Layers",
                hint="Type a neuron name and press Enter to add a chip. "
                     "Each chip = one neuron/layer; use 'A -> B -> C' for connected paths.",
            )
            custom_layer_names = ui.input(
                label="Custom Layer Names (optional, comma-separated)",
                placeholder="e.g., Input, Output",
            ).classes("w-full drocat-input").tooltip(
                "Display names for each neuron layer, in the same order as the chips."
            )
            with param_grid(2):
                dataset = dataset_selector()
                output_dir = dir_input()

            section_header("Appearance", "palette")
            with param_grid(2):
                skeleton_mode = select_input(
                    "Skeleton Mode", SKELETON_MODES, "tube",
                    hint="'tube': 3D tube rendering (detailed). 'line': thin line (fast, for many neurons).",
                )
                legend_mode = select_input(
                    "Legend Mode", ["layer", "type", "single"], "layer",
                    hint="'layer': one legend entry per layer. 'type': per neuron type. 'single': every neuron.",
                )
                neuron_alpha = number_input(
                    "Neuron Opacity", 0.2, 0, 1, 0.1,
                    hint="Transparency of neuron tubes (0=invisible, 1=solid).",
                )
                bg_color = select_input(
                    "Background", ["white", "black"], "white",
                    hint="Background color for the 3D scene and exports.",
                )
                brain_mesh = select_input(
                    "Brain Mesh", BRAIN_MESH_OPTIONS, "template",
                    hint="'template': brain outline. 'whole': full brain surface. 'none': no mesh.",
                )
                vnc_mesh = checkbox_input(
                    "VNC Mesh", False,
                    hint="Show the ventral nerve cord mesh (male-cns / manc datasets).",
                )
            neuron_palette = palette_editor(
                "Neuron Colors",
                value="Category20",
                include_auto=False,
            )

            section_header("Synapses", "bubble_chart")
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

        # ================= Path Network options =================
        with ui.card().classes("w-full drocat-card").props('id="card-path"'):
            section_header("Path Network (PlotPath)", "account_tree")
            ui.label(
                "Load a Find All Paths file (CSV or Excel with path_type / path_bodyId sheets)."
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

            with ui.row().classes("w-full items-center gap-2"):
                with ui.button(icon="upload_file").props("flat dense round").classes(
                    "drocat-upload-trigger"
                ).tooltip("Upload a *_allpaths_info file (CSV/Excel)"):
                    with ui.menu() as path_upload_menu:
                        ui.label("Load path data from Find All Paths").classes(
                            "text-caption drocat-muted px-3 pt-2"
                        )
                        ui.label("CSV / XLSX / XLS with path_type or path_bodyId sheets").classes(
                            "text-caption drocat-muted px-3 pb-1"
                        )
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

            with param_grid(2):
                path_output_dir = dir_input(label="Path Output Directory")
                path_layout = select_input(
                    "Network Layout", NETWORK_LAYOUTS + ["hierarchical"], "hierarchical",
                    hint="Layout algorithm for the HTML network visualization.",
                )
            color_preset = palette_picker(
                "Color Scheme",
                value="Cool",
                catalog=PATH_SCHEMES,
            )
            with ui.row().classes("gap-4"):
                show_path_fig = checkbox_input(
                    "Open in Browser", True,
                    hint="Open the interactive HTML network after rendering.",
                )
            ui.label("Interactive HTML and XLSX connection tables are always generated.").classes(
                "text-caption drocat-muted"
            )

    with results_col:
        output_panel.create(run_label="Generate Visualization", run_icon="play_arrow")

    async def run_visualization():
        output_panel.clear()

        is_3d = "3D Skeleton" in tool_select.value
        colors = COLOR_PRESETS.get(color_preset.get_value(), COLOR_PRESETS["Cool"])

        if is_3d:
            neurons = neuron_chips.value or []
            if not neurons:
                ui.notify("Please add at least one neuron", type="warning")
                return
            rois = roi_select.value or []

            # Per-layer neuron colors sampled from the chosen palette
            neuron_colors = sample_palette(
                neuron_palette.get_colors(), len(neurons)
            )

            # Per-ROI colors so brain regions render independently (Auto = gray)
            mesh_color = (100, 100, 100)
            if roi_palette.get_value() != "Auto (single gray)" and rois:
                mesh_color = sample_palette(roi_palette.get_colors(), len(rois))

            custom_names = [
                n.strip()
                for n in custom_layer_names.value.split(",")
                if n.strip()
            ] if custom_layer_names.value else []

            constructor_params = {
                "dataset": dataset.value,
                "neuron_layers": neurons,
                "custom_layer_names": custom_names,
                "output_dir": output_dir.value,
                "skeleton_mode": skeleton_mode.value,
                "brain_mesh": brain_mesh.value,
                "vnc_mesh": vnc_mesh.value,
                "legend_mode": legend_mode.value,
                "neuron_alpha": float(neuron_alpha.value),
                "neuron_colors": neuron_colors,
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
            tool_name = "plot3d_skeleton"
            output_dir_for_scan = output_dir.value
        else:
            if not path_file_path["path"]:
                ui.notify("Please upload a path file first (Find All Paths output)", type="warning")
                return
            constructor_params = {
                "path_file": path_file_path["path"],
                "output_folder": path_output_dir.value,
                "source_color": colors["source"],
                "intermediate_color": colors["intermediate"],
                "target_color": colors["target"],
                "link_color": colors["link"],
                "network_layout": path_layout.value,
                "showfig": show_path_fig.value,
            }
            method_params = None
            tool_name = "plot_path"
            output_dir_for_scan = path_output_dir.value

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
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir_for_scan)

    output_panel.run_button.on_click(run_visualization)
    output_panel.cancel_button.on_click(runner.cancel)
