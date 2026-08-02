"""NeuronBridge Colabel Tab - Co-labeling analysis for driver lines."""

from nicegui import ui
from ..config import DATASETS
from ..components.common import (
    dataset_selector, neuron_input, number_input, select_input, checkbox_input,
    dir_input, parse_neuron_list, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_nb_colabel_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Co-Labeling Output")

    form_col, results_col = tool_page(
        "Co-Labeling Analysis",
        "Analyze co-labeling patterns between driver lines.",
        icon="layers",
        tag="NeuronBridge",
        doc="nb_colabel.md",
    )

    with form_col:
        with ui.card().classes("w-full"):
            section_header("Driver Lines", "search")
            line_input = neuron_input(label="Driver Line Names", placeholder="e.g., VT000770, SS00001, R10A06")
            with param_grid(2):
                dataset = dataset_selector(label="Dataset (optional)", default=None, datasets=["(all)"] + DATASETS)
                output_dir = dir_input()

        with ui.card().classes("w-full"):
            section_header("Analysis Settings", "tune")
            with ui.row().classes("gap-4"):
                method_jaccard = checkbox_input("Jaccard", True, hint="Binary Jaccard similarity of labeled types.")
                method_weighted = checkbox_input("Weighted Jaccard", True, hint="Score-weighted Jaccard similarity.")
                calc_sparsity = checkbox_input("Line Specificity", True, hint="Calculate sparsity metric per line.")

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                with ui.row().classes("gap-4"):
                    gen_heatmap = checkbox_input("Heatmaps", True, hint="Generate interactive heatmap visualizations.")
                    gen_report = checkbox_input("HTML Report", True, hint="Generate comprehensive HTML analysis report.")
                    visualize_3d = checkbox_input("3D Skeleton", False, hint="Render 3D skeletons of top co-labeled types.")
                viz_top_n = number_input("Visualize Top N Types", 5, 1, 20, hint="Number of top co-labeled types to visualize in 3D.")
                with param_grid(3):
                    top_n_neurons = number_input(
                        "Top N Neurons Per Line", 200, 5, 2000,
                        hint="Maximum neurons considered per driver line for the analysis.",
                    )
                    min_score = number_input(
                        "Min Match Score", 20000, 0, 200000, 1000,
                        hint="Minimum NeuronBridge score for a neuron to be included.",
                    )
                    min_type_avg_score = number_input(
                        "Min Type Avg Score", 10000, 0, 200000, 1000,
                        hint="Minimum average score for a type to be included.",
                    )
                with param_grid(3):
                    sort_by = select_input(
                        "Sort By", ["max_score", "type_avg_score"], "max_score",
                        hint="Sorting key for expression matrices.",
                    )
                    background_color = select_input(
                        "Profile Background", ["white", "black"], "white",
                        hint="Background color for individual profile PDFs.",
                    )
                    pdf_cols = number_input("Profile Images Per Page (cols)", 3, 1, 6)
                    pdf_rows = number_input("Profile Images Per Page (rows)", 2, 1, 6)

    with results_col:
        output_panel.create(run_label="Run Co-Labeling", run_icon="play_arrow")

    async def run_colabel():
        lines = parse_neuron_list(line_input.value)
        if len(lines) < 2:
            ui.notify("Please enter at least two driver lines", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        ds = None if dataset.value == "(all)" else dataset.value

        methods = []
        if method_jaccard.value: methods.append("jaccard")
        if method_weighted.value: methods.append("weighted_jaccard")

        constructor_params = {"verbose": True}

        method_params = {
            "lines": lines,
            "output_dir": output_dir.value,
            "similarity_methods": methods,
            "generate_report": gen_report.value,
            "visualize": gen_heatmap.value,
            "visualize_top_n": int(viz_top_n.value) if visualize_3d.value else 0,
            "top_n_neurons": int(top_n_neurons.value),
            "min_score": float(min_score.value),
            "min_type_avg_score": float(min_type_avg_score.value),
            "sort_by": sort_by.value,
            "background_color": background_color.value,
            "pdf_images_per_page": (int(pdf_cols.value), int(pdf_rows.value)),
        }

        result = await output_panel.run(runner, "nb_colabel", constructor_params, "colabel",
                                        method_params=method_params,
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_colabel)
    output_panel.cancel_button.on_click(runner.cancel)
