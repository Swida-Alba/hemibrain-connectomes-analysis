"""NeuronBridge FindNeuron Tab - Find EM neurons matching LM driver lines."""

from nicegui import ui
from ..config import DEFAULTS, MATCH_ALGORITHMS
from ..components.common import (
    neuron_input, number_input, select_input, checkbox_input,
    dir_input, parse_neuron_list, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner


def create_nb_find_neuron_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("EM Neurons Output")

    form_col, results_col = tool_page(
        "Find EM Neurons",
        "Find EM neurons matching GAL4 driver line morphology.",
        icon="search",
    )

    with form_col:
        with ui.card().classes("w-full"):
            section_header("Driver Line Query", "search")
            line_input = neuron_input(label="Driver Line Names", placeholder="e.g., LH173, VT037867, SS00731")
            output_dir = dir_input()

        with ui.card().classes("w-full"):
            section_header("Search Parameters", "tune")
            with param_grid(2):
                match_algo = select_input("Algorithm", MATCH_ALGORITHMS, DEFAULTS["match_algorithm"])
                top_n = number_input("Top N Results", 20, 5, 100)

            # --- Advanced Settings (collapsed) ---
            with ui.expansion("Advanced Settings", icon="settings_suggest").classes("w-full"):
                with param_grid(2):
                    visualize = checkbox_input("Visualize Top Neurons", True, hint="Generate 3D skeleton visualizations of matched neurons.")
                    viz_top_n = number_input("Visualize Top N", 10, 1, 50, hint="Number of top types/bodyIds to render in 3D.")
                generate_pdf = checkbox_input("PDF Summary", True, hint="Generate PDF/PPTX with individual neuron profiles.")

    with results_col:
        output_panel.create(run_label="Find EM Neurons", run_icon="play_arrow")

    async def run_find_neuron():
        lines = parse_neuron_list(line_input.value)
        if not lines:
            ui.notify("Please enter at least one driver line", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {"verbose": True}

        method_params = {
            "line_names": lines,
            "output_dir": output_dir.value,
            "match_type": match_algo.value,
            "top_n": int(top_n.value),
            "visualize_top_n": int(viz_top_n.value) if visualize.value else 0,
            "generate_individual_profiles": ['pdf'] if generate_pdf.value else None,
        }

        result = await output_panel.run(runner, "nb_find_neuron", constructor_params, "find_neurons",
                                        method_params=method_params,
                                        output_dir=output_dir.value)

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], output_dir.value)

    output_panel.run_button.on_click(run_find_neuron)
    output_panel.cancel_button.on_click(runner.cancel)
