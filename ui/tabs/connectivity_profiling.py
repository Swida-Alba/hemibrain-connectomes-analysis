"""ConnectivityProfiling Tab - Intra-dataset connectivity profile comparison."""

from nicegui import ui
from ..config import DEFAULTS
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input, checkbox_input,
    dir_input, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_connectivity_profiling_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Profiling Output")

    form_col, results_col = tool_page(
        "Connectivity Profiling",
        "Compare connectivity profiles within a single dataset.",
        icon="analytics",
        doc="connectivity_profiling.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Query Neurons", "search")
            query_input = neuron_list_input(
                label="Neurons to Compare",
                placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, aMe10, aMe9)",
                hint="Enter 2+ neurons to compare profiles. Upload CSV/TSV/Excel for large lists.",
            )
            with param_grid(2):
                dataset = dataset_selector(hint="Dataset to compute profiles from.")
                output_dir = dir_input()

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
                    "All four similarity matrices are generated: Jaccard, cosine, "
                    "rank correlation, and rank-correlation union."
                ).classes("text-caption drocat-muted")
                cluster_heatmap = checkbox_input(
                    "Generate Heatmaps", True,
                    hint="Create the interactive Ward-clustered heatmap files.",
                )
                with param_grid(3):
                    min_synapse_threshold = number_input(
                        "Min Synapse Threshold", 3, 1, 100,
                        hint="Minimum synapse count for a connection to enter a profile.",
                    )
                    aggregation_level = select_input(
                        "Aggregation Level", ["type", "bodyid"], "type",
                        hint="'type': aggregate profiles per neuron type. 'bodyid': per individual neuron.",
                    )
                    skip_bodyid_level = select_input(
                        "BodyId-Level Computation", ["auto", "skip", "compute"], "auto",
                        hint="'auto': skip when >1000 bodyIds. 'skip': type-level only. "
                             "'compute': always compute bodyId-level matrices.",
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

    with results_col:
        output_panel.create(run_label="Run Profiling", run_icon="play_arrow")

    async def run_profiling():
        mode, neurons = query_input.get_value()
        query = apply_filter_mode(neurons, mode)
        if not query:
            ui.notify("Please enter at least one neuron", type="warning")
            return

        skip_bodyid_param = {
            "auto": "auto",
            "skip": True,
            "compute": False,
        }.get(skip_bodyid_level.value, "auto")

        output_panel.clear()
        output_panel.set_running(True)

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

        constructor_params = {
            "query": query,
            "dataset": dataset.value,
            "output_dir": output_dir.value,
            "top_k": int(top_k.value),
            "top_m": int(top_m.value),
            "min_synapse_threshold": int(min_synapse_threshold.value),
            "direction": direction,
            "generate_heatmaps": cluster_heatmap.value,
            "show_figures": show_figures.value,
            "verbose": True,
            "use_cache": True,
            "aggregation_level": aggregation_level.value,
            "skip_bodyId_level": skip_bodyid_param,
            "ensure_cache_complete": full_cache.value,
        }

        result = await output_panel.run(runner, "connectivity_profiling", constructor_params, "run",
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_profiling)
    output_panel.cancel_button.on_click(runner.cancel)
