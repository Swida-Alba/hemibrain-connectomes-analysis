"""Regression tests for analysis-tab skeleton visualization settings."""

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.profile_comparator import HomologFinder  # noqa: E402
from neuronbridge_finder import NeuronBridgeFinder  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402
from visualization_options import (  # noqa: E402
    default_analysis_skeleton_mesh_simplification,
)


def test_homolog_visualization_defaults_to_template_brain():
    finder = HomologFinder(verbose=False)
    assert finder.visualization_settings["brain_mesh"] == "template"


def test_homolog_visualization_settings_override_renderer_defaults():
    finder = HomologFinder(
        verbose=False,
        visualization_settings={
            "brain_mesh": "none",
            "skeleton_mode": "line",
            "show_fig": True,
        },
    )
    options = finder._homolog_visualizer_kwargs(
        {"brain_mesh": "template", "show_fig": False},
        dataset="test:v1",
        neuron_layers=["A"],
    )
    assert options["brain_mesh"] == "none"
    assert options["skeleton_mode"] == "line"
    assert options["show_fig"] is True
    assert options["skeleton_mesh_simplification"] == 0.95


def test_analysis_skeleton_simplification_defaults_are_dataset_aware():
    assert default_analysis_skeleton_mesh_simplification("male-cns:v1.0") == 0.95
    assert default_analysis_skeleton_mesh_simplification("hemibrain:v1.2.1") == 0.95
    assert default_analysis_skeleton_mesh_simplification("flywire_FAFB_v783") == 0.98
    assert default_analysis_skeleton_mesh_simplification("FAFB:v783") == 0.98


def test_visualization_warning_notes_record_effective_simplification(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.skeleton_mesh_simplification = 0.98
    visualizer.skeleton_mode = "tube"
    visualizer.save_folder = str(tmp_path)
    visualizer._vprint = lambda *args, **kwargs: None

    visualizer._write_user_warning_notes()

    notes = (tmp_path / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "skeleton_mesh_simplification=0.98" in notes
    assert "rendering only" in notes


def test_html_simplification_warning_uses_dataset_specific_thresholds():
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.90
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "tube", 0.901
    )["threshold"] == 0.90

    assert VisualizeSkeleton._skeleton_simplification_warning(
        "flywire_FAFB_v783", "flywire", "tube", 0.95
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "flywire_FAFB_v783", "flywire", "tube", 0.951
    )["threshold"] == 0.95

    # The warning is for the tube-surface pipeline, not line rendering or the
    # separate BANC mesh pipeline.
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "male-cns:v1.0", "neuprint", "line", 0.99
    ) is None
    assert VisualizeSkeleton._skeleton_simplification_warning(
        "flywire_BANC_v888", "flywire", "tube", 0.99
    ) is None


def test_html_writer_embeds_plotly_runtime_and_injects_warning(tmp_path):
    class FakeFigure:
        def __init__(self):
            self.kwargs = None

        def write_html(self, path, **kwargs):
            self.kwargs = kwargs
            Path(path).write_text(
                "<html><head></head><body><div id='figure'></div></body></html>",
                encoding="utf-8",
            )

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "male-cns:v1.0"
    visualizer.client_type = "neuprint"
    visualizer.skeleton_mode = "tube"
    visualizer.skeleton_mesh_simplification = 0.95
    visualizer._vprint = lambda *args, **kwargs: None

    figure = FakeFigure()
    path = tmp_path / "skeleton.html"
    visualizer._write_plotly_html(figure, str(path))

    assert figure.kwargs["include_plotlyjs"] is True
    html = path.read_text(encoding="utf-8")
    assert "drocat-skeleton-simplification-warning" in html
    assert "skeleton_mesh_simplification=0.950" in html
    assert "fixed simp90 skeleton cache" in html
    assert html.index("drocat-skeleton-simplification-warning") > html.index("<body>")


def test_real_skeleton_html_is_portable_without_plotly_sidecar(tmp_path):
    import plotly.graph_objects as go

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.client_type = "flywire"
    visualizer.skeleton_mode = "tube"
    visualizer.skeleton_mesh_simplification = 0.98
    visualizer._vprint = lambda *args, **kwargs: None

    path = tmp_path / "portable.html"
    visualizer._write_plotly_html(
        go.Figure(go.Scatter3d(x=[0], y=[0], z=[0])), str(path)
    )

    html = path.read_text(encoding="utf-8")
    assert 'src="plotly.min.js"' not in html
    assert "Plotly.newPlot" in html
    assert "drocat-skeleton-simplification-warning" in html


def test_all_skeleton_result_paths_accept_visualization_settings():
    assert "visualization_settings" in inspect.signature(
        NeuronBridgeFinder.find_neurons_batch
    ).parameters
    assert "visualization_settings" in inspect.signature(
        NeuronBridgeFinder.analyze_colabeling
    ).parameters
