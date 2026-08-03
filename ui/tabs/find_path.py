"""
FindPath Tab - Multi-hop pathfinding between neuron groups.
"""

from nicegui import ui

from ..config import DEFAULTS, PATHFINDING_ALGORITHMS, FILTER_OPTIONS, OUTPUT_FORMATS, NETWORK_LAYOUTS
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_find_path_tab():
    """Create the FindPath tab UI."""
    runner = ScriptRunner()
    output_panel = OutputPanel("Pathfinding Output")

    form_col, results_col = tool_page(
        "Find All Paths",
        "Discover multi-hop pathways between source and target neuron groups.",
        icon="route",
        doc="find_path.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Neuron Selection", "hub")
            with param_grid(2):
                source_input = neuron_list_input(
                    label="Source Neurons",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, aMe10)",
                    hint="Enter neuron types, bodyIds, or patterns. Upload CSV/TSV/Excel for large lists.",
                )
                target_input = neuron_list_input(
                    label="Target Neurons",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., PPL101, DN1p)",
                    hint="Enter neuron types, bodyIds, or patterns. Upload CSV/TSV/Excel for large lists.",
                )
            with param_grid(2):
                dataset = dataset_selector(
                    hint="Select the connectome dataset.",
                    allow_custom=True,
                )
                output_dir = dir_input()

        with ui.card().classes("w-full drocat-card"):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                max_interlayer = number_input(
                    "Max Intermediate Layers", DEFAULTS["max_interlayer"], 0, 10,
                    hint="Maximum number of intermediate neuron layers between source and target. Higher = more paths but slower.",
                )
                min_synapse = number_input(
                    "Min Synapse Count", DEFAULTS["min_synapse_num"], 1, 100,
                    hint="Minimum number of synapses for a connection to be included. Filters out weak/noisy connections.",
                )
                edge_limit = number_input(
                    "Edge Limit", DEFAULTS["edgeN_limit"], 10, 5000,
                    hint="Maximum number of edges to consider per neuron. Limits memory usage for highly connected neurons.",
                )

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                with param_grid(2):
                    custom_source_name = ui.input(
                        label="Custom Source Name (optional)",
                        placeholder="e.g., aMe_clock",
                    ).classes("w-full").tooltip("Custom label for source group in output files/plots.")
                    custom_target_name = ui.input(
                        label="Custom Target Name (optional)",
                        placeholder="e.g., PPL1_dopamine",
                    ).classes("w-full").tooltip("Custom label for target group in output files/plots.")

                keyword_filter = ui.input(
                    label="Keywords to Exclude from Paths (comma-separated)",
                    placeholder="e.g., None, unknown",
                ).classes("w-full").tooltip("Paths containing these keywords in neuron types will be removed.")

                with param_grid(3):
                    min_ratio = number_input(
                        "Min Connection Ratio", DEFAULTS["min_ratio"], 0, 1, 0.01,
                        hint="Minimum weight/post ratio (0-1). Higher = stronger connections only. 0 = include all.",
                    )
                    min_traversal = number_input(
                        "Min Traversal Prob.", DEFAULTS["min_traversal_probability"], 0, 1, 0.01,
                        hint="Minimum traversal probability (ratio/0.3, capped at 1.0). Controls path confidence threshold.",
                    )
                    pathfinding_algo = select_input(
                        "Algorithm", PATHFINDING_ALGORITHMS, DEFAULTS["pathfinding"],
                        hint="Bidirectional: fastest for short paths. DP: prunes dead ends. MemoizedDFS: best for deep paths. DFS: lowest memory.",
                    )

                with param_grid(3):
                    filter_by = select_input(
                        "Filter By", FILTER_OPTIONS, DEFAULTS["filter_by"],
                        hint="'bodyId': filter at individual neuron level. 'type': aggregate by neuron type.",
                    )
                    output_format = select_input(
                        "Output Format", OUTPUT_FORMATS, DEFAULTS["output_format"],
                        hint="'csv': faster, smaller. 'xlsx': Excel format with formatting.",
                    )
                    network_layout = select_input(
                        "Network Layout", NETWORK_LAYOUTS, DEFAULTS["network_layout"],
                        hint="Layout algorithm for the HTML network visualization.",
                    )

                with ui.row().classes("gap-4"):
                    use_cache = checkbox_input(
                        "Use Cache", DEFAULTS["use_cache"],
                        hint="Cache neuron data locally for 10-100x speedup on repeated runs.",
                    )
                    skip_bodyid = checkbox_input(
                        "Skip BodyId in Output", True,
                        hint="Exclude individual bodyId-level results. Only show type-level aggregation.",
                    )
                    show_fig = checkbox_input(
                        "Show Figure", False,
                        hint="Open the interactive HTML visualization automatically after completion.",
                    )
                    cache_only = checkbox_input(
                        "Cache Only (Offline)", False,
                        hint="Use only local cache and never contact the server. "
                             "Requires the cache to be pre-built.",
                    )
                with param_grid(2):
                    saveas = ui.input(
                        label="Save Folder Name (optional)",
                        placeholder="e.g., aMe_clock_paths",
                    ).classes("w-full drocat-input").tooltip(
                        "Custom output folder name. Leave empty for the unified auto name "
                        "(findpath_<dataset>_<src>_to_<tgt>_<params>_<timestamp>)."
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
                    find_reciprocal = checkbox_input(
                        "Find Reciprocal Connections", False,
                        hint="Enrich the path graph with reciprocal direct connections.",
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
        output_panel.create(run_label="Find All Paths", run_icon="account_tree")

    async def run_pathfinding():
        # Get values from neuron_list_input (returns (mode, list))
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()

        sources = apply_filter_mode(src_neurons, src_mode)
        targets = apply_filter_mode(tgt_neurons, tgt_mode)

        if not sources:
            ui.notify("Please enter at least one source neuron", type="warning")
            return
        if not targets:
            ui.notify("Please enter at least one target neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        # Parse keyword filter
        kw_raw = keyword_filter.value.strip()
        keywords = [k.strip() for k in kw_raw.split(",") if k.strip()] if kw_raw else ['None']

        constructor_params = {
            "dataset": dataset.value,
            "sourceNeurons": sources,
            "targetNeurons": targets,
            "output_dir": output_dir.value,
            "min_synapse_num": int(min_synapse.value),
            "min_ratio": float(min_ratio.value),
            "min_traversal_probability": float(min_traversal.value),
            "max_interlayer": int(max_interlayer.value),
            "filter_by": filter_by.value,
            "pathfinding": pathfinding_algo.value,
            "network_layout": network_layout.value,
            "use_cache": use_cache.value,
            "edgeN_limit": int(edge_limit.value),
            "output_format": output_format.value,
            "skip_bodyId": skip_bodyid.value,
            "showfig": show_fig.value,
            "custom_source_name": custom_source_name.value or '',
            "custom_target_name": custom_target_name.value or '',
            "keyword_in_path_to_remove": keywords,
            "cache_only": cache_only.value,
            "saveas": saveas.value.strip() or "",
            "separate_hemispheres": separate_hemi.value,
            "keep_only_hemisphere_conserved_connections": keep_hemi_conserved.value,
            "symmetry_analysis": symmetry_analysis.value,
            "find_reciprocal": find_reciprocal.value,
        }

        result = await output_panel.run(runner, "find_path", constructor_params, "find_all_path",
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_pathfinding)
    output_panel.cancel_button.on_click(runner.cancel)
