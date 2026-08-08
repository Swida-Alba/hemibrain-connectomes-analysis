"""Cross-Dataset Comparison Tab - runs ComparisonAnalyzer over N datasets."""

from nicegui import ui
from ..config import COMPARISON_MODES, PATHFINDING_ALGORITHMS, SEARCH_COLUMNS
from ..components.common import (
    dataset_multi_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.mapping_editor import mapping_selector, selected_mapping_file_path
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_inter_dataset_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Comparison Output")

    form_col, results_col = tool_page(
        "Cross-Dataset Comparison",
        "Compare connectivity across multiple datasets (2+).",
        icon="sync_alt",
        doc="cross_dataset.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Datasets", "storage")
            datasets_select = dataset_multi_selector(
                label="Datasets to compare (select 2+)",
                default=["male-cns:v0.9", "hemibrain:v1.2.1"],
            )
            nicknames_input = neuron_list_input(
                label="Nicknames (optional)",
                show_filter=False,
                show_upload=False,
                hint="Short display names for each dataset (same order). Leave empty for auto.",
            )

        with ui.card().classes("w-full drocat-card"):
            section_header("Neuron Selection", "hub")
            source_input = neuron_list_input(
                label="Source Neurons",
                placeholder="Type or upload CSV/TSV/Excel with neuron types/bodyIds",
                hint="Source neurons for pathfinding. Upload a CSV/TSV/Excel file (first column) or type comma-separated.",
            )
            target_input = neuron_list_input(
                label="Target Neurons",
                placeholder="Type or upload CSV/TSV/Excel with neuron types/bodyIds",
                hint="Target neurons for pathfinding. Upload a CSV/TSV/Excel file (first column) or type comma-separated.",
            )
            output_dir = dir_input()
            mapping_select = mapping_selector()

        with ui.card().classes("w-full drocat-card"):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                comparison_mode = select_input(
                    "Mode", COMPARISON_MODES, "path",
                    hint="'path': discover edges via path traversal. 'edge': compare edges independently by weight.",
                )
                max_interlayer = number_input(
                    "Max Intermediate Layers", 2, 0, 5,
                    hint="Maximum hops between source and target.",
                )
                thresholds_input = neuron_list_input(
                    label="Synapse Thresholds",
                    initial=[3, 5, 10],
                    show_filter=False,
                    show_upload=False,
                    hint="List of min synapse thresholds to analyze. "
                         "Type one threshold per chip (e.g. 3, 5, 10), or keep the defaults.",
                )

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                with param_grid(2):
                    pathfinding = select_input(
                        "Pathfinding Algorithm", PATHFINDING_ALGORITHMS, "MemoizedDFS",
                        hint="MemoizedDFS: recommended default (fastest measured at all depths, no graph copy). DFS: backward memoized, best with few targets. MeetInMiddle: shallow queries. DP: robust. Bidirectional: shortest-first but high memory.",
                        help_doc="pathfinding_algorithms.html",
                    )
                    top_edges = number_input(
                        "Top Edges", 500, 10, 5000,
                        hint="Maximum edges to include in analysis per threshold.",
                    )
                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, "auto",
                    hint="Which columns to search when resolving neuron names in every dataset. "
                         "'auto': all columns (bodyId -> type -> instance -> flywireType/others). "
                         "Use 'type'/'instance'/'bodyId' to restrict the search.",
                )
                with ui.row().classes("gap-4"):
                    skip_bodyid = checkbox_input("Skip BodyId Level", True, hint="Skip bodyId-level results for speed.")
                    cache_only = checkbox_input("Cache Only (Offline)", False, hint="Use only local cache, no server connection.")
                    auto_type_mapping = checkbox_input("Auto Type Mapping", False, hint="Auto-map type names across datasets.")
                with param_grid(3):
                    min_ratio = number_input(
                        "Min Connection Ratio", 0.0, 0, 1, 0.01,
                        hint="Minimum weight/post ratio for an edge to be kept.",
                    )
                    min_prob = number_input(
                        "Min Traversal Prob.", 0.0, 0, 1, 0.01,
                        hint="Minimum traversal probability for an edge to be kept.",
                    )
                    output_format = select_input(
                        "Output Format", ["csv", "xlsx"], "csv",
                        hint="Format for exported data tables.",
                    )
                with param_grid(2):
                    parallel = checkbox_input(
                        "Parallel Processing", True,
                        hint="Run per-dataset work in parallel where possible.",
                    )
                    max_workers = number_input(
                        "Max Workers", 4, 1, 16,
                        hint="Number of parallel workers (only used when Parallel Processing is on).",
                    )

        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-hemisphere"'):
            section_header("Hemisphere Analysis", "sync_alt")
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
                    "Symmetry Analysis", True,
                    hint="Generate per-dataset hemisphere symmetry summaries (auto-enabled with Separate Hemispheres).",
                )
                find_reciprocal = checkbox_input(
                    "Find Reciprocal Connections", False,
                    hint="Build reciprocal graphs and include them in reports.",
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
        output_panel.create(run_label="Run Comparison", run_icon="play_arrow")

    async def run_comparison():
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()
        sources = apply_filter_mode(src_neurons, src_mode)
        targets = apply_filter_mode(tgt_neurons, tgt_mode)

        if not sources:
            ui.notify("Please provide at least one source neuron", type="warning")
            return

        datasets = datasets_select.value or []
        if len(datasets) < 2:
            ui.notify("Please add at least 2 datasets to compare", type="warning")
            return

        # Parse thresholds (chip values are already normalized to integers;
        # split comma-joined chips defensively in case a list was typed into
        # one chip before the run).
        try:
            thresholds = [
                int(v)
                for item in thresholds_input.get_value()[1]
                for v in str(item).replace(' ', '').split(',')
                if v
            ]
        except (TypeError, ValueError):
            ui.notify("Invalid thresholds format. Use comma-separated integers.", type="negative")
            return
        if not thresholds:
            ui.notify("Please enter at least one synapse threshold", type="warning")
            return

        # Parse nicknames
        nicknames = [str(n) for n in nicknames_input.get_value()[1]] or None
        if nicknames and len(nicknames) != len(datasets):
            ui.notify(f"Nickname count ({len(nicknames)}) must match dataset count ({len(datasets)})", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {
            "datasets": datasets,
            "source_neurons": sources,
            "target_neurons": targets,
            "output_folder": output_dir.value,
            "comparison_mode": comparison_mode.value,
            "max_interlayer": int(max_interlayer.value),
            "thresholds": thresholds,
            "top_edges": int(top_edges.value),
            "pathfinding": pathfinding.value,
            "search_columns": search_columns.value,
            "skip_bodyId": skip_bodyid.value,
            "cache_only": cache_only.value,
            "auto_type_mapping": auto_type_mapping.value,
            "_min_ratio": float(min_ratio.value),
            "_min_prob": float(min_prob.value),
            "_output_format": output_format.value,
            "parallel": parallel.value,
            "max_workers": int(max_workers.value) if parallel.value else None,
            "separate_hemispheres": separate_hemi.value,
            "keep_only_hemisphere_conserved_connections": keep_hemi_conserved.value,
            "symmetry_analysis": symmetry_analysis.value,
            "find_reciprocal": find_reciprocal.value,
        }
        if nicknames:
            constructor_params["datasets_nickname"] = nicknames
        mapping_path = selected_mapping_file_path(mapping_select.value)
        if mapping_path:
            constructor_params["overall_mapping_json"] = mapping_path

        result = await output_panel.run(runner, "inter_dataset", constructor_params, "run",
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_comparison)
    output_panel.cancel_button.on_click(runner.cancel)
