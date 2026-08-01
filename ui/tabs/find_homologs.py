"""FindHomologs Tab - Cross-dataset homolog finding."""

from nicegui import ui
from ..config import DEFAULTS, DATASETS, SIMILARITY_METRICS
from ..components.common import (
    dataset_selector, neuron_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_find_homologs_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Homolog Output")

    form_col, results_col = tool_page(
        "Homolog Finding",
        "Find potential homologs across datasets via connectivity profile similarity.",
        icon="compare",
    )

    with form_col:
        with ui.card().classes("w-full"):
            section_header("Source Neuron", "search")
            source_input = neuron_input(
                label="Source Neuron (type or bodyId)",
                placeholder="e.g., aMe12",
                hint="Single neuron type or bodyId to find cross-dataset homologs for.",
            )
            with param_grid(2):
                source_dataset = dataset_selector(label="Source Dataset", hint="Dataset where the source neuron lives.")
                target_dataset = dataset_selector(label="Target Dataset", default=DATASETS[1], hint="Dataset to search for homologs in.")
            output_dir = dir_input()

        with ui.card().classes("w-full"):
            section_header("Search Parameters", "tune")
            with param_grid(3):
                top_n = number_input("Top N Candidates", DEFAULTS["top_n"], 5, 100, hint="Number of top homolog candidates to return.")
                top_k = number_input("Top K Partners", DEFAULTS["top_k"], 5, 50, hint="Top K partners per direction for profile construction.")
                top_m = number_input("Min Types (M)", DEFAULTS["top_m"], 3, 20, hint="Minimum unique partner types in profile.")
            with param_grid(2):
                similarity_metric = select_input("Similarity Metric", SIMILARITY_METRICS, DEFAULTS["similarity_metric"], hint="Metric for comparing connectivity profiles.")
                viz_top_n = number_input("Visualize Top N", 5, 1, 20, hint="Number of top candidates to render as 3D skeletons.")
            with ui.row().classes("gap-4"):
                use_fast = checkbox_input("Fast Search", True, hint="Use adjacency expansion for faster candidate discovery.")
                vector_prefilter = checkbox_input("Vector Pre-filtering", True, hint="Pre-filter candidates using vector cosine similarity.")
                expand_2hop = checkbox_input("2-Hop Expansion", True, hint="Include 2-hop typed partners for untyped 1-hop neurons.")
                visualize = checkbox_input("Visualize Candidates", True, hint="Generate 3D skeleton visualizations of top matches.")
                full_cache = checkbox_input(
                    "Pre-build Full Dataset Cache", False,
                    hint="Fetch connections for EVERY uncached neuron before searching. "
                         "Very slow on first use (can take hours); leave off to fetch only "
                         "the connections the search needs.",
                )

    with results_col:
        output_panel.create(run_label="Find Homologs", run_icon="play_arrow")

    async def run_homologs():
        source = source_input.value.strip()
        if not source:
            ui.notify("Please enter a source neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

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
            "visualize_top_n": int(viz_top_n.value),
            "ensure_cache_complete": full_cache.value,
        }

        method_name = "find_homologs_fast" if use_fast.value else "find_homologs"
        result = await runner.run("find_homologs", constructor_params, method_name,
                                  log_callback=output_panel.log, output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], output_dir.value)

    output_panel.run_button.on_click(run_homologs)
    output_panel.cancel_button.on_click(runner.cancel)
