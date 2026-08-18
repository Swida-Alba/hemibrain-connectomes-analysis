"""Regression tests for compact skeleton transformation progress output."""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class CaptureProgressBar:
    """Minimal outer-bar stub that records postfix refreshes."""

    def __init__(self):
        self.statuses = []

    def set_postfix_str(self, status, refresh=True):
        self.statuses.append((status, refresh))


def make_neuron(name):
    neuron = navis.TreeNeuron(pd.DataFrame({
        "node_id": np.array([0], dtype=np.int64),
        "parent_id": np.array([-1], dtype=np.int64),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "radius": np.array([1.0]),
        "type": ["root"],
    }))
    neuron.name = name
    return neuron


def make_mesh(body_id=42):
    vertices = np.array([
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
    ])
    faces = np.array([
        (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
    ], dtype=np.int64)
    mesh = navis.MeshNeuron({"vertices": vertices, "faces": faces})
    mesh.id = body_id
    return mesh


def test_transform_progress_reuses_outer_bar(monkeypatch, capsys):
    """Successful transforms refresh one bar instead of printing status lines."""
    monkeypatch.setitem(sys.modules, "flybrains", types.ModuleType("flybrains"))
    monkeypatch.setattr(
        vs_mod.navis.transforms.registry,
        "shortest_bridging_seq",
        lambda source, target: ([source, target], ["step"]),
    )
    monkeypatch.setattr(vs_mod.navis, "xform", lambda neuron, transform: neuron)

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.verbose = True
    progress = CaptureProgressBar()

    result = visualizer._xform_neurons_safe(
        navis.NeuronList([make_neuron("left"), make_neuron("right")]),
        source="raw",
        target="template",
        layer_label="Layer 1 (example)",
        progress_bar=progress,
    )

    assert len(result) == 2
    statuses = [status for status, _ in progress.statuses]
    assert statuses[0].endswith("2 pts | raw→template")
    assert any("current=left" in status for status in statuses)
    assert any("current=right" in status for status in statuses)
    assert statuses[-1].endswith("done 0.0s")

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_fafb_visualizer_api_bypass_fetches_mesh_without_skeletonizing(
        tmp_path, monkeypatch):
    """The uncached FAFB render path must stay mesh-native."""
    import cave_data_fetcher as cave

    calls = []

    class FakeCaveFetcher:
        def __init__(self, *args, **kwargs):
            assert kwargs["project_root"] == str(tmp_path)

        def fetch_mesh(self, body_id, use_cache=False):
            calls.append((body_id, use_cache))
            return make_mesh(body_id)

    monkeypatch.setattr(cave, "CAVEDataFetcher", FakeCaveFetcher)

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.script_path = str(tmp_path)
    visualizer.verbose = False
    visualizer.cache_neurons = False
    visualizer._flywire_skeleton_access = {"cave_token": "test-token"}

    result = visualizer._fetch_fafb_skeletons_via_api(
        [42], cache_prepared=False, soma_positions={"42": [0, 0, 0]})

    assert calls == [(42, False)]
    assert isinstance(result["42"], navis.MeshNeuron)


def test_fafb_visualizer_extrusion_repair_forces_mesh_refresh(
        tmp_path, monkeypatch):
    """Extrusion replacement must refresh even when prepared caching is on."""
    import cave_data_fetcher as cave

    calls = []

    class FakeCaveFetcher:
        def __init__(self, *args, **kwargs):
            assert kwargs["project_root"] == str(tmp_path)

        def fetch_fafb_meshes(self, body_ids, **kwargs):
            calls.append((list(body_ids), kwargs))
            return navis.NeuronList([make_mesh(body_ids[0])])

    monkeypatch.setattr(cave, "CAVEDataFetcher", FakeCaveFetcher)

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.script_path = str(tmp_path)
    visualizer.verbose = False
    visualizer.cache_neurons = True
    visualizer._flywire_skeleton_access = {"cave_token": "test-token"}

    result = visualizer._fetch_fafb_skeletons_via_api(
        [42],
        cache_prepared=True,
        force_refresh=True,
        soma_positions={"42": [0, 0, 0]},
    )

    assert isinstance(result["42"], navis.MeshNeuron)
    assert calls[0][0] == [42]
    assert calls[0][1]["use_cache"] is True
    assert calls[0][1]["force_refresh"] is True
