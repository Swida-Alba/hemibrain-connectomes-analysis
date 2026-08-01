"""NeuronBridge FindLines Tab - Find LM driver lines matching EM neurons."""

from nicegui import ui
from ..config import DEFAULTS, DATASETS, MATCH_ALGORITHMS
from ..components.common import (
    dataset_selector, advanced_neuron_input, number_input, select_input,
    checkbox_input, dir_input, parse_neuron_list, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_nb_find_lines_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Driver Lines Output")

    form_col, results_col = tool_page(
        "Find Driver Lines",
        "Find GAL4 / Split-GAL4 driver lines matching EM neurons.",
        icon="biotech",
    )

    with form_col:
        with ui.card().classes("w-full"):
            section_header("Query Neurons", "search")
            query_input = advanced_neuron_input(
                label="EM Neurons (bodyId, type, or instance)",
                placeholder="e.g., aMe12, 720575940610453042",
                hint="Enter EM neuron identifiers. Use filter mode for pattern matching across types.",
            )
            with param_grid(2):
                dataset = dataset_selector(
                    label="Dataset (optional)", default=None, datasets=["(all)"] + DATASETS,
                    hint="Restrict search to one dataset, or select (all) to search everywhere.",
                )
                output_dir = dir_input()

        with ui.card().classes("w-full"):
            section_header("Search Parameters", "tune")
            with param_grid(2):
                match_algo = select_input(
                    "Algorithm", MATCH_ALGORITHMS, DEFAULTS["match_algorithm"],
                    hint="'cds': Color Depth Search (fast). 'pppm': Point Pattern (precise). 'both': run both.",
                )
                top_n_gal4 = number_input("Top N Lines", 20, 5, 100, hint="Max driver lines to return per query.")

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                top_n_split = number_input("Top N Split-GAL4 (separate)", 20, 5, 100, hint="Max Split-GAL4 lines when separated.")
                separate_split = checkbox_input(
                    "Separate Split-GAL4 Results", True,
                    hint="Generate separate summary CSVs for GAL4/LexA vs Split-GAL4 lines.",
                )
                ui.separator()
                ui.label("Image Download").classes("text-caption font-bold")
                with ui.row().classes("gap-4"):
                    download_images = checkbox_input("Download Images", False, hint="Download matched images from NeuronBridge.")
                    download_flylight = checkbox_input("From FlyLight", False, hint="Download from FlyLight S3/CDN.")
                    generate_pdf = checkbox_input("PDF Summary", True, hint="Create a PDF with downloaded images ordered by score.")

    with results_col:
        output_panel.create(run_label="Find Driver Lines", run_icon="play_arrow")

    async def run_find_lines():
        mode, text = query_input.get_value()
        query = apply_filter_mode(parse_neuron_list(text), mode)
        if not query:
            ui.notify("Please enter at least one neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        ds = None if dataset.value == "(all)" else dataset.value

        constructor_params = {
            "verbose": True,
            "separate_splitgal4": separate_split.value,
        }

        # Determine download_images source
        dl_source = None
        if download_images.value and download_flylight.value:
            dl_source = 'both'
        elif download_flylight.value:
            dl_source = 'flylight'
        elif download_images.value:
            dl_source = 'neuronbridge'

        method_params = {
            "queries": query,
            "dataset": ds,
            "output_dir": output_dir.value,
            "match_type": match_algo.value,
            "download_images": dl_source,
            "download_img_for_top_n_lines": int(top_n_gal4.value) if dl_source else None,
            "summary_format": 'pdf' if generate_pdf.value else None,
        }

        result = await output_panel.run(runner, "nb_find_lines", constructor_params, "find_lines",
                                        method_params=method_params,
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_find_lines)
    output_panel.cancel_button.on_click(runner.cancel)
