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


def test_all_skeleton_result_paths_accept_visualization_settings():
    assert "visualization_settings" in inspect.signature(
        NeuronBridgeFinder.find_neurons_batch
    ).parameters
    assert "visualization_settings" in inspect.signature(
        NeuronBridgeFinder.analyze_colabeling
    ).parameters
