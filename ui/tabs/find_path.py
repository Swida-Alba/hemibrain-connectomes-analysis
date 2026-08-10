"""
FindPath Tab - Multi-hop pathfinding between neuron groups.
"""

from nicegui import ui

from ..config import DEFAULTS, PATHFINDING_ALGORITHMS, FILTER_OPTIONS, OUTPUT_FORMATS, NETWORK_LAYOUTS, SEARCH_COLUMNS
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.mapping_editor import mapping_selector, selected_mapping_file_path
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
                mapping_select = mapping_selector(label="Custom Grouping")

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-core"'):
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
            interlayer_warning = ui.label(
                "⚠️ Layers ≥ 4: the path count grows combinatorially (branching^depth) — "
                "reconstruction can take hours and produce billions of paths. Raise "
                "Min Synapse Count / Min Connection Ratio / Min Traversal Prob., or "
                "tighten the Graph Edge Limit in Advanced Settings."
            ).classes("text-caption text-amber-8").set_visibility(False)
            max_interlayer.on_value_change(
                lambda e: interlayer_warning.set_visibility((e.value or 0) >= 4)
            )

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                keyword_filter = neuron_list_input(
                    label="Keywords to Exclude from Paths",
                    show_filter=False,
                    show_upload=False,
                    hint="Paths containing these keywords in neuron types will be removed. "
                         "Type a keyword and press Enter (or leave the field) to add it as a chip.",
                )

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
                        hint="MemoizedDFS: recommended default (fastest measured at all depths, no graph copy). DFS: backward memoized, best with few targets. MeetInMiddle: shallow queries. DP: robust. Bidirectional: shortest-first but high memory.",
                        help_doc="pathfinding_algorithms.html",
                    )
                with ui.row().classes("gap-4"):
                    limit_edges = checkbox_input(
                        "Limit Graph Edges (strongest only)", True,
                        hint="Keep only the strongest connections of the discovered "
                             "network graph before pathfinding: the top-N edges by "
                             "synapse weight (types/custom groups: 1000, bodyIds: 5000). "
                             "This bounds the combinatorial path count (branching^depth) "
                             "and focuses on strong connections. Uncheck = complete "
                             "graph with ALL edges (can be very slow for deep layers).",
                    )
                    edge_limit_groups = number_input(
                        "Edge Limit – Types/Groups", 1000, 100, 100000000,
                        hint="Top-N strongest edges kept in the type-level and "
                             "custom-group-level graphs. 0 = unlimited.",
                    )
                    edge_limit_bodyid = number_input(
                        "Edge Limit – BodyIds", 5000, 100, 100000000,
                        hint="Top-N strongest edges kept in the bodyId-level graph. "
                             "0 = unlimited.",
                    )
                    edge_limit_groups.set_enabled(False)
                    edge_limit_bodyid.set_enabled(False)
                    visualize_early = checkbox_input(
                        "Visualize Network Before Reconstruction", False,
                        hint="Draw the discovered network graph (all weighted edges) into "
                             "network_early/ right after the layers are fetched — before the "
                             "path enumeration — so you see the explored topology immediately "
                             "while deep reconstruction is still running.",
                    )
                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, "auto",
                    hint="Which columns to search when resolving neuron names. "
                         "'auto': all columns (bodyId -> type -> instance -> flywireType/others). "
                         "Use 'type'/'instance'/'bodyId' to restrict the search.",
                )
                filter_by = select_input(
                    "Filter By", FILTER_OPTIONS, DEFAULTS["filter_by"],
                    hint="'bodyId': filter at individual neuron level. 'type': aggregate by neuron type.",
                )

                with ui.row().classes("gap-4"):
                    use_cache = checkbox_input(
                        "Use Cache", DEFAULTS["use_cache"],
                        hint="Cache neuron data locally for 10-100x speedup on repeated runs.",
                    )
                    cache_only = checkbox_input(
                        "Cache Only (Offline)", False,
                        hint="Use only local cache and never contact the server. "
                             "Requires the cache to be pre-built.",
                    )

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-output"'):
            section_header("Output Options", "output")
            with param_grid(2):
                custom_source_name = ui.input(
                    label="Custom Source Name (optional)",
                    placeholder="e.g., aMe_clock",
                ).classes("w-full").tooltip("Custom label for source group in output files/plots.")
                custom_target_name = ui.input(
                    label="Custom Target Name (optional)",
                    placeholder="e.g., PPL1_dopamine",
                ).classes("w-full").tooltip("Custom label for target group in output files/plots.")
            with param_grid(3):
                output_format = select_input(
                    "Output Format", OUTPUT_FORMATS, DEFAULTS["output_format"],
                    hint="'csv': faster, smaller. 'xlsx': Excel format with formatting.",
                )
                network_layout = select_input(
                    "Network Layout", NETWORK_LAYOUTS, DEFAULTS["network_layout"],
                    hint="Layout algorithm for the HTML network visualization.",
                )
                saveas = ui.input(
                    label="Save Folder Name (optional)",
                    placeholder="e.g., aMe_clock_paths",
                ).classes("w-full drocat-input").tooltip(
                    "Custom output folder name. Leave empty for the unified auto name "
                    "(findallpath_<dataset>_<src>_to_<tgt>_<params>_<timestamp>; "
                    "findpath_... in per-path mode)."
                )
            with ui.row().classes("gap-4"):
                skip_bodyid = checkbox_input(
                    "Skip BodyId in Output", True,
                    hint="Exclude individual bodyId-level results. Only show type-level aggregation.",
                )
                show_fig = checkbox_input(
                    "Show Figure", False,
                    hint="Open the interactive HTML visualization automatically after completion.",
                )

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-hemisphere"'):
            section_header("Hemisphere Analysis", "sync_alt")
            with ui.row().classes("gap-4"):
                separate_hemi = checkbox_input(
                    "Separate Hemispheres (L/R)", False,
                    hint="Split type/group aggregation into _L/_R/_U hemisphere labels.",
                )
                hemi_filter = select_input(
                    "Hemisphere", ["both", "left", "right"], "both",
                    hint="'both': all neurons. 'left'/'right': restrict to that hemisphere. "
                         "Neurons WITHOUT an explicit hemisphere (no _L/_R instance suffix "
                         "or Soma side) are always included in every option.",
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
                    hemi_filter.set_enabled(True)
                else:
                    keep_hemi_conserved.disable()
                    symmetry_analysis.disable()
                    hemi_filter.set_enabled(False)
            separate_hemi.on_value_change(lambda _e: _sync_hemisphere_options())
            _sync_hemisphere_options()

    with results_col:
        output_panel.create(run_label="Find All Paths", run_icon="account_tree")

    async def run_pathfinding():
        # Get values from neuron_list_input (returns (mode, list))
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()

        def _sync_edge_limits_enabled():
            edge_limit_groups.set_enabled(limit_edges.value)
            edge_limit_bodyid.set_enabled(limit_edges.value)

        limit_edges.on_value_change(lambda _e: _sync_edge_limits_enabled())

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

        # Parse keyword filter (chips are already individual keywords)
        keywords = [str(k) for k in keyword_filter.get_value()[1]] or ['None']

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
            "graph_edge_limit_groups": (
                int(edge_limit_groups.value) if limit_edges.value else 0
            ),
            "graph_edge_limit_bodyid": (
                int(edge_limit_bodyid.value) if limit_edges.value else 0
            ),
            "visualize_before_reconstruct": visualize_early.value,
            "search_columns": search_columns.value,
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
            "hemisphere_filter": hemi_filter.value,
            "keep_only_hemisphere_conserved_connections": keep_hemi_conserved.value,
            "symmetry_analysis": symmetry_analysis.value,
            "find_reciprocal": find_reciprocal.value,
        }
        mapping_path = selected_mapping_file_path(mapping_select.value)
        if mapping_path:
            constructor_params["custom_mapping_file"] = mapping_path

        result = await output_panel.run(runner, "find_path", constructor_params, "find_all_path",
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_pathfinding)
    output_panel.cancel_button.on_click(runner.cancel)
