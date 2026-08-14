"""FindHomologs Tab - Cross-dataset homolog finding."""

from nicegui import ui
from ..config import DEFAULTS, DATASETS, SIMILARITY_METRICS
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions


def create_find_homologs_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Homolog Output")
    source_dataset = None

    def _type_suggest(text):
        dataset_name = source_dataset.value if source_dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    form_col, results_col = tool_page(
        "Homolog Finding",
        "Find potential homologs across datasets via connectivity profile similarity.",
        icon="compare",
        doc="find_homologs.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Source Neuron", "search")
            source_input = neuron_list_input(
                label="Source Neuron (type or bodyId)",
                show_filter=False,
                show_upload=False,
                max_items=1,
                suggestions=_type_suggest,
                available_neurons=lambda: source_dataset.value
                if source_dataset is not None else "",
                hint="Single neuron type or bodyId to find cross-dataset homologs for. Only one value is accepted.",
            )
            with param_grid(2):
                source_dataset = dataset_selector(label="Source Dataset", hint="Dataset where the source neuron lives.")
                target_dataset = dataset_selector(label="Target Dataset", default=DATASETS[1], hint="Dataset to search for homologs in.")
            output_dir = dir_input(scope="find_homologs")

        with ui.card().classes("w-full drocat-card"):
            section_header("Search Parameters", "tune")
            with param_grid(3):
                top_n = number_input("Top N Candidates", DEFAULTS["top_n"], 5, 100, hint="Number of top homolog candidates to return.")
                top_k = number_input("Top K Partners", DEFAULTS["top_k"], 5, 50, hint="Top K partners per direction for profile construction.")
                top_m = number_input("Min Types (M)", DEFAULTS["top_m"], 3, 20, hint="Minimum unique partner types in profile.")
            with param_grid(2):
                similarity_metric = select_input(
                    "Sort By", SIMILARITY_METRICS, DEFAULTS["similarity_metric"],
                    hint="Metric used ONLY for ordering the candidate list (top-N cut). "
                         "All similarity metrics (jaccard, cosine, rank_corr, combined) are "
                         "always computed — same backend as Connectivity Profiling.",
                )
            with ui.row().classes("gap-4"):
                use_fast = checkbox_input("Fast Search", True, hint="Use adjacency expansion for faster candidate discovery.")
                vector_prefilter = checkbox_input("Vector Pre-filtering", True, hint="Pre-filter candidates using vector cosine similarity.")
                expand_2hop = checkbox_input("2-Hop Expansion", True, hint="Include 2-hop typed partners for untyped 1-hop neurons.")
            with param_grid(3):
                min_synapse_threshold = number_input(
                    "Min Synapse Threshold", 3, 1, 100,
                    hint="Minimum synapse count for a connection to enter a profile.",
                )
                use_cache = checkbox_input(
                    "Use Cache", True,
                    hint="Cache profiles and connections locally for faster repeat searches.",
                )
                use_auto_type_mapping = checkbox_input(
                    "Auto Type Mapping", True,
                    hint="Standardize partner type names to canonical (male-cns) names "
                         "before cross-dataset comparison.",
                )
            with ui.row().classes("w-full items-center gap-4"):
                visualize = checkbox_input(
                    "Visualize Candidates",
                    True,
                    hint="Generate 3D skeleton visualizations of the top matches.",
                )
                visualization_settings = skeleton_visualization_settings(
                    default_top_n=5,
                    top_n_label="Visualize Top N Candidates",
                    top_n_hint="Number of top homolog candidates to render as 3D skeletons.",
                    default_visualize_by="type",
                    dataset_provider=lambda: [
                        source_dataset.value,
                        target_dataset.value,
                    ],
                    dataset_watchers=[source_dataset, target_dataset],
                )
            saveas = ui.input(
                label="Save Folder Name (optional)",
                placeholder="e.g., aMe12_homologs",
            ).classes("w-full drocat-input").tooltip(
                "Custom output folder name. Leave empty for the unified auto name "
                "(findhomologs_<source_ds>_to_<target_ds>_<query>_<timestamp>)."
            )
            full_cache = checkbox_input(
                "Pre-build Full Dataset Cache", False,
                hint="Fetch connections for EVERY uncached neuron before searching. "
                     "Very slow on first use (can take hours); leave off to fetch only "
                     "the connections the search needs.",
            )

    with results_col:
        output_panel.create(run_label="Find Homologs", run_icon="play_arrow")

    async def run_homologs():
        source_vals = source_input.get_value()[1]
        source = str(source_vals[0]).strip() if source_vals else ""
        if not source:
            ui.notify("Please enter a source neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)
        visualization_values = visualization_settings.values()

        constructor_params = {
            "source": source,
            "source_dataset": source_dataset.value,
            "target_dataset": target_dataset.value,
            "output_dir": output_dir.value,
            "top_n": int(top_n.value),
            "top_k": int(top_k.value),
            "top_m": int(top_m.value),
            "similarity_metric": similarity_metric.value,
            "vector_prefiltering": vector_prefilter.value,
            "include_untyped_partners": expand_2hop.value,
            "visualize_skeleton": visualize.value,
            "visualize_top_n": (
                visualization_values["visualize_top_n"]
                if visualize.value else 0
            ),
            "visualization_settings": visualization_values,
            "min_synapse_threshold": int(min_synapse_threshold.value),
            "use_cache": use_cache.value,
            "saveas": saveas.value.strip() or "",
            "use_auto_type_mapping": use_auto_type_mapping.value,
            "ensure_cache_complete": full_cache.value,
        }

        method_name = "find_homologs_fast" if use_fast.value else "find_homologs"
        result = await output_panel.run(runner, "find_homologs", constructor_params, method_name,
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_homologs)
    output_panel.cancel_button.on_click(runner.cancel)
