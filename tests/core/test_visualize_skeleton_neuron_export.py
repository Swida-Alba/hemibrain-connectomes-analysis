"""Tests for the compact neuron-info export used by 3D skeleton views."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def test_neuron_info_export_merges_layers_without_roi_counts(tmp_path):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.save_folder = str(tmp_path)
    visualizer.saveas = "male-cns_v1_0"
    visualizer.layer_names = ["query_l-LNv_x1", "r1_aMe12_x2"]
    visualizer.neuron_dfs = [
        pd.DataFrame({"bodyId": [101], "type": ["l-LNv"]}),
        pd.DataFrame({"bodyId": [201, 202], "instance": ["aMe12", "aMe12"]}),
    ]

    output_path = visualizer._save_neuron_info_csv()

    assert output_path == str(tmp_path / "male-cns_v1_0_neuron_info.csv")
    exported = pd.read_csv(output_path)
    assert list(exported.columns) == ["bodyId", "type", "viz_layer", "instance"]
    assert exported["viz_layer"].tolist() == [
        "query_l-LNv_x1",
        "r1_aMe12_x2",
        "r1_aMe12_x2",
    ]
    assert exported["bodyId"].tolist() == [101, 201, 202]
    assert not (tmp_path / "male-cns_v1_0_neuron_info.xlsx").exists()
    assert "roi_count" not in exported.columns
