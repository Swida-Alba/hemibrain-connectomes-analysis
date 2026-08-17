"""Tests for the merged inter-layer synapse export."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def _visualizer(tmp_path, output_format):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.save_folder = str(tmp_path)
    visualizer.saveas = "demo"
    visualizer.output_format = output_format
    visualizer._vprint = lambda *args, **kwargs: None
    return visualizer


def _synapse_frames():
    first = pd.DataFrame({"bodyId_pre": [1], "bodyId_post": [2]})
    first["viz_layer"] = "0->1"
    second = pd.DataFrame({"bodyId_pre": [3], "bodyId_post": [4]})
    second["viz_layer"] = "1->2"
    return [first, second]


def test_synapse_export_merges_layers_into_one_xlsx_sheet(tmp_path):
    visualizer = _visualizer(tmp_path, "xlsx")

    output_path = visualizer._save_synapse_data(_synapse_frames())

    assert output_path == str(tmp_path / "demo_synapses.xlsx")
    workbook = pd.ExcelFile(output_path)
    assert workbook.sheet_names == ["synapses"]
    exported = pd.read_excel(output_path, sheet_name="synapses")
    assert exported["viz_layer"].tolist() == ["0->1", "1->2"]
    assert exported["bodyId_pre"].tolist() == [1, 3]


def test_synapse_export_defaults_to_global_csv_format(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.save_folder = str(tmp_path)
    visualizer.saveas = "default"
    visualizer._vprint = lambda *args, **kwargs: None

    output_path = visualizer._save_synapse_data(_synapse_frames())

    assert output_path == str(tmp_path / "default_synapses.csv")
    assert not (tmp_path / "default_synapses.xlsx").exists()


def test_synapse_export_merges_layers_into_one_csv(tmp_path):
    visualizer = _visualizer(tmp_path, "csv")

    output_path = visualizer._save_synapse_data(_synapse_frames())

    assert output_path == str(tmp_path / "demo_synapses.csv")
    exported = pd.read_csv(output_path)
    assert exported["viz_layer"].tolist() == ["0->1", "1->2"]
    assert exported["bodyId_post"].tolist() == [2, 4]
