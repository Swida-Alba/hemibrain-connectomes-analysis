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
