"""ConnectivityProfiling Tab - Intra-dataset connectivity profile comparison."""

from nicegui import ui
from ..config import DEFAULTS, get_tab_output_dir
from ..components.common import (
    dataset_multi_selector, neuron_list_input, number_input, select_input, checkbox_input,
    dir_input, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.mapping_editor import custom_grouping_block
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import datasets_suggestions


def _resolve_profiling_output_dir(value):
    """Resolve the path selected in the profiling tab before a run starts."""
    selected = str(value or "").strip()
    return selected or str(get_tab_output_dir("connectivity_profiling")).strip()


def create_connectivity_profiling_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Profiling Output")
    datasets_select = None

    def _type_suggest(text):
        selected = datasets_select.value if datasets_select is not None else []
        return datasets_suggestions(text, selected, limit=None)

    form_col, results_col = tool_page(
        "Connectivity Profiling",
        "Compare connectivity profiles within and across selected datasets.",
        icon="analytics",
        doc="connectivity_profiling.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card").props('id="card-profiling-datasets"'):
            section_header("Datasets", "storage")
            datasets_select = dataset_multi_selector(
                label="Datasets to compare (select one or more)",
                default=["male-cns:v1.0"],
                hint="Select one or more datasets. One dataset with multiple thresholds "
                     "is also supported. Two or more datasets profile the same query in "
                     "each dataset (names mapped per dataset) and add within-dataset "
                     "(intra) plus across-dataset (inter, same neuron) comparisons. "
                     "The inter-dataset overview puts all queried neurons in rows and "
                     "dataset pairs in columns.",
            )
            output_dir = dir_input(scope="connectivity_profiling")

        with ui.card().classes("w-full drocat-card").props('id="card-profiling-neurons"'):
            section_header("Query Neurons", "search")
            query_input = neuron_list_input(
                label="Neurons to Compare",
                placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, aMe10, aMe9)",
                hint="Enter 2+ neurons to compare profiles. Upload CSV/TSV/Excel for large lists.",
                suggestions=_type_suggest,
                available_neurons=lambda: list(datasets_select.value or [])
                if datasets_select is not None else [],
            ).classes("drocat-fixed-neuron-input")

        with ui.card().classes("w-full drocat-card"):
            section_header("Profile Construction", "build")
            with param_grid(2):
                top_k = number_input(
                    "Top K Partners", DEFAULTS["top_k"], 5, 50,
                    hint="Number of top synaptic partners per direction to include in the profile.",
                )
                top_m = number_input(
                    "Min Unique Types (M)", DEFAULTS["top_m"], 3, 20,
                    hint="Minimum unique partner types. If top_k yields fewer, K is expanded.",
                )

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                with ui.row().classes("gap-4"):
                    analyze_upstream = checkbox_input("Upstream", True, hint="Include presynaptic (input) partners in profile.")
                    analyze_downstream = checkbox_input("Downstream", True, hint="Include postsynaptic (output) partners in profile.")

                ui.separator()
                ui.label(
                    "All six similarity matrices are generated: overall, Jaccard, "
                    "weighted Jaccard, cosine, rank correlation, and rank-correlation "
                    "union (same metric set as the Find Homologs tab). Overall combines "
                    "upstream and downstream connectivity."
                ).classes("text-caption drocat-muted")
                cluster_heatmap = checkbox_input(
                    "Generate Heatmaps", True,
                    hint="Create VisPath heatmaps for editing and Plotly heatmaps in the report.",
                )
                with param_grid(3):
                    min_synapse_threshold = number_input(
                        "Min Synapse Threshold", 3, 1, 100,
                        hint="Minimum synapse count for a connection to enter a profile.",
                    )
                    aggregation_level = select_input(
                        "Aggregation Level", ["type", "bodyid", "custom group"], "type",
                        hint="'type': each matched neuron type is one row — patterns like "
                             "'aMe.*' or name-filter inputs expand into their independent "
                             "types. 'bodyid': every individual neuron is one row. "
                             "'custom group': rows come from the LabelMapper preset below.",
                    ).props('id=select-aggregation')
                    skip_bodyid_level = select_input(
                        "BodyId-Level Computation", ["auto", "skip", "compute"], "auto",
                        hint="'auto': skip bodyId matrices only when >1000 bodyIds. "
                             "'skip': type-level only. 'compute': always include bodyId "
                             "and type-average-bodyId matrices. Type and bodyId levels "
                             "are both available in the profiling output.",
                    )
                with ui.row().classes("gap-4"):
                    show_figures = checkbox_input(
                        "Show Figures", False,
                        hint="Open generated heatmaps in the browser.",
                    )
                full_cache = checkbox_input(
                    "Pre-build Full Dataset Cache", False,
                    hint="Fetch connections for EVERY uncached neuron before profiling. "
                         "Very slow on first use (can take hours); leave off to use the "
                         "connections already cached.",
                )

            # Custom grouping via LabelMapper presets (only for the
            # 'custom group' aggregation level)
            custom_group_box = ui.card().classes("w-full drocat-card").props('id=card-custom-group')
            with custom_group_box:
                section_header("Custom Groups (LabelMapper)", "group_work")
                mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                    label="Custom Grouping Preset",
                    hint="Saved LabelMapper preset (manage in the Settings tab) or inline "
                         "groups. Each source-side group becomes one row of the comparison "
                         "matrix.",
                    tab_key="profiling",
                    datasets_provider=lambda: list(datasets_select.value or []),
                    watch_elements=[datasets_select],
                    query_inputs={"query": query_input},
                )
                ui.label(
                    "Groups are read from the preset's source mapping: each custom label "
                    "is one group, and its members for the selected datasets fill the rows. "
                    "The neuron query above is ignored in this mode."
                ).classes("text-caption drocat-muted")
            custom_group_box.set_visibility(False)

            aggregation_level.on_value_change(
                lambda e: custom_group_box.set_visibility(
                    (e.value or "") == "custom group"
                )
            )

    with results_col:
        output_panel.create(run_label="Run Profiling", run_icon="play_arrow")

    async def run_profiling():
        mode, neurons = query_input.get_value()
        query = apply_filter_mode(neurons, mode)
        if not query:
            ui.notify("Please enter at least one neuron", type="warning")
            return

        selected_datasets = list(datasets_select.value or [])
        if not selected_datasets:
            ui.notify("Select one or more datasets", type="warning")
            return

        skip_bodyid_param = {
            "auto": "auto",
            "skip": True,
            "compute": False,
        }.get(skip_bodyid_level.value, "auto")

        # Custom-group mode needs a mapping (preset or inline); resolve it
        # before the running state so an invalid board aborts cleanly.
        mapping_path = None
        if aggregation_level.value == "custom group":
            mapping_path, mapping_ok = resolve_grouping()
            if not mapping_ok:
                return
            if not mapping_path:
                ui.notify(
                    "Select a LabelMapper preset or define inline groups for "
                    "the custom groups", type="warning")
                return

        # Determine direction from checkboxes
        if analyze_upstream.value and analyze_downstream.value:
            direction = 'both'
        elif analyze_upstream.value:
            direction = 'upstream'
        elif analyze_downstream.value:
            direction = 'downstream'
        else:
            ui.notify("Select upstream, downstream, or both", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        # Keep the path selected in this tab as the single source of truth for
        # both the backend constructor and the output-file scanner.  The
        # fallback matters when the input has not emitted its first browser
        # change event yet (for example, after opening the tab and clicking
        # Run immediately).
        profiling_output_dir = _resolve_profiling_output_dir(output_dir.value)

        constructor_params = {
            "query": query,
            "datasets": selected_datasets,
            "output_dir": profiling_output_dir,
            "top_k": int(top_k.value),
            "top_m": int(top_m.value),
            "min_synapse_threshold": int(min_synapse_threshold.value),
            "direction": direction,
            "generate_heatmaps": cluster_heatmap.value,
            "show_figures": show_figures.value,
            "verbose": True,
            "use_cache": True,
            "aggregation_level": {
                "type": "type",
                "bodyid": "bodyid",
                "custom group": "custom",
            }[aggregation_level.value],
            "skip_bodyId_level": skip_bodyid_param,
            "ensure_cache_complete": full_cache.value,
        }

        if aggregation_level.value == "custom group":
            constructor_params["custom_mapping_file"] = mapping_path

        result = await output_panel.run(
            runner,
            "connectivity_profiling",
            constructor_params,
            "run",
            output_dir=profiling_output_dir,
        )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(
            result["files"], result.get("output_folder") or profiling_output_dir
        )

    output_panel.run_button.on_click(run_profiling)
    output_panel.cancel_button.on_click(runner.cancel)
