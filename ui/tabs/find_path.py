"""
FindPath Tab - Multi-hop pathfinding between neuron groups.
"""

from nicegui import ui

from ..config import DEFAULTS, PATHFINDING_ALGORITHMS, FILTER_OPTIONS, OUTPUT_FORMATS, NETWORK_LAYOUTS, SEARCH_COLUMNS
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.mapping_editor import custom_grouping_block
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions


def create_find_path_tab():
    """Create the FindPath tab UI."""
    runner = ScriptRunner()
    output_panel = OutputPanel("Pathfinding Output")
    dataset = None
    search_columns = None

    def _type_suggest(text):
        """Auto-suggest from the selected dataset's type names. Type matches
        come first for string input; the range expands to instance/bodyId only
        when no type matched and the search scope is 'auto'."""
        ds = dataset.value if dataset is not None else ""
        scope = search_columns.value if search_columns is not None else "auto"
        # Keep the complete candidate pool for local continuation filtering;
        # neuron_list_input applies the display limit after it narrows the
        # pool, so valid names beyond the first page remain reachable.
        return dataset_suggestions(text, ds, scope, limit=None)

    form_col, results_col = tool_page(
        "Complete Paths",
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
                    suggestions=_type_suggest,
                    available_neurons=lambda: dataset.value if dataset is not None else "",
                )
                target_input = neuron_list_input(
                    label="Target Neurons",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., PPL101, DN1p)",
                    hint="Enter neuron types, bodyIds, or patterns. Upload CSV/TSV/Excel for large lists.",
                    suggestions=_type_suggest,
                    available_neurons=lambda: dataset.value if dataset is not None else "",
                )
            with param_grid(2):
                dataset = dataset_selector(
                    hint="Select the connectome dataset.",
                    allow_custom=True,
                )
                output_dir = dir_input(scope="find_path")
                mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                    label="Custom Grouping",
                    tab_key="find_path",
                    datasets_provider=lambda: [dataset.value] if dataset.value else [],
                    watch_elements=[dataset],
                    query_inputs={"source": source_input, "target": target_input},
                )

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-core"'):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                max_interlayer = number_input(
                    "Max Intermediate Layers", DEFAULTS["max_interlayer"], 0, None,
                    hint="Maximum number of intermediate neuron layers between source and target. Higher = more paths but slower.",
                )
                min_synapse = number_input(
                    "Min Synapse Count", DEFAULTS["min_synapse_num"], 1, 100,
                    hint="Minimum number of synapses for a connection to be included. Filters out weak/noisy connections.",
                )
                edge_limit = number_input(
                    "Visualization Edge Limit", DEFAULTS["edgeN_limit"], 10, 5000,
                    hint="Maximum edges drawn per visualization (network / Sankey / heatmap, "
                         "including the network_early preview). Limits memory usage for highly "
                         "connected neurons.",
                )
            interlayer_warning = ui.label(
                "⚠️ Layers ≥ 4: the path count grows combinatorially (branching^depth) — "
                "reconstruction can take hours and produce billions of paths. Raise "
                "Min Synapse Count / Min Connection Ratio / Min Traversal Prob., "
                "tighten the Graph Edge Limit in Advanced Settings, or minimize/"
                "batch the source and target sets."
            ).classes("text-caption text-amber-8").set_visibility(False)
            edge_limit_bodyid_hint = None

            def _on_max_interlayer_change(e):
                interlayer_warning.set_visibility((e.value or 0) >= 4)
                # the bodyId edge limit only applies to deep searches
                enabled = (e.value or 0) >= 3
                edge_limit_bodyid.set_enabled(enabled)
                if edge_limit_bodyid_hint is not None:
                    edge_limit_bodyid_hint.set_visibility(not enabled)

            max_interlayer.on_value_change(_on_max_interlayer_change)

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
                    with ui.column().classes("gap-0"):
                        edge_limit_bodyid = number_input(
                            "Edge Limit – BodyIds", 1000000, 0, 1000000000,
                            hint="Top-N strongest non-reserved edges kept in the bodyId-level "
                                 "graph (source/target edges are always kept in addition). "
                                 "Applied only when Layers ≥ 3 (deep searches, where the path "
                                 "count grows combinatorially); shallow runs keep the complete "
                                 "graph. Type-level and custom-group paths are derived from the "
                                 "discovered bodyId paths and need no edge limit. "
                                 "0 = unlimited (can be very slow for deep layers).",
                        )
                        edge_limit_bodyid_hint = ui.label(
                            "Unavailable for shallow searches (max intermediate layers 0–2); "
                            "set Max Intermediate Layers to 3+ to enable BodyId edge trimming."
                        ).classes("text-caption text-grey-7").set_visibility(
                            (max_interlayer.value or 0) < 3
                        )
                    # enabled only for deep searches (max_interlayer >= 3)
                    edge_limit_bodyid.set_enabled((max_interlayer.value or 0) >= 3)
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
                    "(find-paths-complete_<dataset>_<src>_to_<tgt>_<params>_<timestamp>)."
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
                    "Hemisphere-aware", False,
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
                    hint="Keep only edges conserved between hemispheres (requires Hemisphere-aware).",
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
                    # dependent options are unchecked AND disabled so a
                    # greyed-out True never reaches the backend
                    keep_hemi_conserved.value = False
                    symmetry_analysis.value = False
                    hemi_filter.value = 'both'
                    keep_hemi_conserved.disable()
                    symmetry_analysis.disable()
                    hemi_filter.set_enabled(False)
            separate_hemi.on_value_change(lambda _e: _sync_hemisphere_options())
            _sync_hemisphere_options()

    with results_col:
        output_panel.create(run_label="Complete Paths", run_icon="account_tree")

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

        # Resolve custom grouping first: an invalid inline board aborts the
        # run before the output panel enters its running state.
        mapping_path, mapping_ok = resolve_grouping()
        if not mapping_ok:
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
            "graph_edge_limit_bodyid": int(edge_limit_bodyid.value),
            # Complete Paths no longer exposes the early network preview in
            # the UI; keep the backend behavior explicitly disabled.
            "visualize_before_reconstruct": False,
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
        if mapping_path:
            constructor_params["custom_mapping_file"] = mapping_path

        result = await output_panel.run(runner, "find_path", constructor_params, "find_all_path",
                                        output_dir=output_dir.value)

        # InitializeNeuronInfo runs before the path search and reports the
        # resolved source/target counts. Record only after that confirmation;
        # unknown queries that resolve to zero neurons never enter history.
        match_info = result.get("neuron_match") or {}
        if match_info.get("any_pair"):
            from ..history_store import record as _record_history
            _record_history(
                [str(v) for v in src_neurons + tgt_neurons],
                datasets=[dataset.value] if dataset.value else [],
            )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_pathfinding)
    output_panel.cancel_button.on_click(runner.cancel)
