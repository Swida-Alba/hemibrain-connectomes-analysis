"""FindDirect Tab - Direct connection analysis."""

from nicegui import ui
from ..config import DEFAULTS, FILTER_OPTIONS, OUTPUT_FORMATS, NETWORK_LAYOUTS
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_find_direct_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Direct Connections Output")

    form_col, results_col = tool_page(
        "Direct Connections",
        "Find direct synaptic connections between neuron groups.",
        icon="arrow_forward",
        doc="find_direct.md",
    )

    with form_col:
        with ui.card().classes("w-full"):
            section_header("Neuron Selection", "hub")
            with param_grid(2):
                source_input = neuron_list_input(
                    label="Source Neurons",
                    placeholder="Type or upload CSV/XLSX (e.g., aMe.*)",
                    hint="Enter source neuron types/bodyIds or upload file. Leave empty for all neurons.",
                )
                target_input = neuron_list_input(
                    label="Target Neurons (optional)",
                    placeholder="Leave empty to find all downstream targets",
                    hint="Enter target neurons or upload file. Leave empty for all connections from sources.",
                )
            with param_grid(2):
                dataset = dataset_selector(
                    hint="Select the connectome dataset to query.",
                )
                output_dir = dir_input()

        with ui.card().classes("w-full"):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                min_synapse = number_input(
                    "Min Synapse Count", DEFAULTS["min_synapse_num"], 1, 100,
                    hint="Minimum synapses for a connection to be included.",
                )
                edge_limit = number_input(
                    "Edge Limit", 50, 10, 5000,
                    hint="Maximum edges to include in output. Limits result size.",
                )

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                with param_grid(3):
                    min_ratio = number_input(
                        "Min Connection Ratio", DEFAULTS["min_ratio"], 0, 1, 0.01,
                        hint="Minimum weight/post ratio. Higher = stronger connections only.",
                    )
                    min_traversal = number_input(
                        "Min Traversal Prob.", DEFAULTS["min_traversal_probability"], 0, 1, 0.01,
                        hint="Minimum traversal probability threshold.",
                    )
                    filter_by = select_input(
                        "Filter By", FILTER_OPTIONS, DEFAULTS["filter_by"],
                        hint="'bodyId': individual level. 'type': aggregated by type.",
                    )
                with param_grid(3):
                    output_format = select_input(
                        "Output Format", OUTPUT_FORMATS, DEFAULTS["output_format"],
                        hint="'csv' or 'xlsx'.",
                    )
                    network_layout = select_input(
                        "Network Layout", NETWORK_LAYOUTS, DEFAULTS["network_layout"],
                        hint="Layout for the HTML network graph.",
                    )
                    use_cache = checkbox_input(
                        "Use Cache", DEFAULTS["use_cache"],
                        hint="Enable local caching for faster repeated queries.",
                    )
                exclude_intra = checkbox_input(
                    "Exclude Intra-type Connections", False,
                    hint="Remove connections between neurons of the same type.",
                )
                with param_grid(2):
                    custom_source_name = ui.input(
                        label="Custom Source Name (optional)",
                        placeholder="e.g., aMe_clock",
                    ).classes("w-full drocat-input").tooltip(
                        "Custom label for the source group in output files."
                    )
                    custom_target_name = ui.input(
                        label="Custom Target Name (optional)",
                        placeholder="e.g., PPL1_dopamine",
                    ).classes("w-full drocat-input").tooltip(
                        "Custom label for the target group in output files."
                    )
                with param_grid(2):
                    saveas = ui.input(
                        label="Save Folder Name (optional)",
                        placeholder="e.g., aMe_to_PPL_direct",
                    ).classes("w-full drocat-input").tooltip(
                        "Custom output folder name. Leave empty for the unified auto name."
                    )
                    cache_only = checkbox_input(
                        "Cache Only (Offline)", False,
                        hint="Use only local cache and never contact the server.",
                    )

                ui.separator()
                with ui.row().classes("gap-4"):
                    separate_hemi = checkbox_input(
                        "Separate Hemispheres (L/R)", False,
                        hint="Split type/group aggregation into _L/_R/_U hemisphere labels.",
                    )
                    keep_hemi_conserved = checkbox_input(
                        "Keep Only Hemisphere-Conserved Edges", False,
                        hint="Keep only edges conserved between hemispheres (requires Separate Hemispheres).",
                    )
                    symmetry_analysis = checkbox_input(
                        "Symmetry Analysis", False,
                        hint="Generate ipsilateral vs contralateral symmetry outputs.",
                    )
                def _sync_hemisphere_options():
                    if separate_hemi.value:
                        keep_hemi_conserved.enable()
                        symmetry_analysis.enable()
                    else:
                        keep_hemi_conserved.disable()
                        symmetry_analysis.disable()
                separate_hemi.on_value_change(lambda _e: _sync_hemisphere_options())
                _sync_hemisphere_options()

    with results_col:
        output_panel.create(run_label="Find Direct Connections", run_icon="play_arrow")

    async def run_direct():
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()

        sources = apply_filter_mode(src_neurons, src_mode)
        targets = apply_filter_mode(tgt_neurons, tgt_mode)

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {
            "dataset": dataset.value,
            "sourceNeurons": sources if sources else [],
            "targetNeurons": targets,
            "output_dir": output_dir.value,
            "min_synapse_num": int(min_synapse.value),
            "min_ratio": float(min_ratio.value),
            "min_traversal_probability": float(min_traversal.value),
            "filter_by": filter_by.value,
            "network_layout": network_layout.value,
            "use_cache": use_cache.value,
            "edgeN_limit": int(edge_limit.value),
            "output_format": output_format.value,
            "exclude_intra_type_connections": exclude_intra.value,
            "custom_source_name": custom_source_name.value.strip() or "",
            "custom_target_name": custom_target_name.value.strip() or "",
            "saveas": saveas.value.strip() or "",
            "cache_only": cache_only.value,
            "separate_hemispheres": separate_hemi.value,
            "keep_only_hemisphere_conserved_connections": keep_hemi_conserved.value,
            "symmetry_analysis": symmetry_analysis.value,
        }

        result = await output_panel.run(runner, "find_direct", constructor_params, "find_direct",
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_direct)
    output_panel.cancel_button.on_click(runner.cancel)
