"""Tests for line-mode CAVE skeletonization at the render boundary.

Covers `_process_fafb_layer` line/tube behavior:
- MeshNeuron sources are skeletonized only for line rendering
- tube mode never skeletonizes (CAVE meshes stay mesh-native)
- the skeletonized product is in-memory only and never written to any
  cache (raw SWC cache, prepared mesh cache)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def make_neuron(n_nodes=2000, body_id="42"):
    types = (["root"] + ["slab"] * max(0, n_nodes - 2) + ["end"])
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n_nodes - 1)),
                               dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * 1000.0,
        "y": np.zeros(n_nodes),
        "z": np.zeros(n_nodes),
        "radius": np.ones(n_nodes),
        "type": types[:n_nodes],
    })
    nrn = navis.TreeNeuron(nodes)
    nrn.soma = None
    nrn.id = body_id
    return nrn


def make_mesh(body_id="99"):
    import trimesh
    mesh = navis.MeshNeuron(
        trimesh.creation.icosphere(subdivisions=3))
    mesh.id = body_id
    return mesh


def build_line_visualizer(monkeypatch):
    """Stubbed visualizer recording every skeletonize / cache-write call."""
    calls = {"skeletonize": 0}

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.skeleton_mode = "line"
    visualizer._vprint = lambda *args, **kwargs: None

    real_skeletonize = navis.skeletonize

    def recorded_skeletonize(mesh):
        calls["skeletonize"] += 1
        return real_skeletonize(mesh)

    monkeypatch.setattr(vs_mod.navis, "skeletonize", recorded_skeletonize)

    def fail_on_write(*args, **kwargs):
        raise AssertionError(
            "line-mode skeletonization must never write a cache")

    visualizer._save_cached_neurons = fail_on_write
    visualizer._persist_fetched_neuprint_skeletons = fail_on_write
    return visualizer, calls


class TestLineModeSkeletonization:
    def test_mesh_source_skeletonized_only_for_line_rendering(
            self, monkeypatch):
        visualizer, calls = build_line_visualizer(monkeypatch)
        mesh = make_mesh()

        out, done = visualizer._process_fafb_layer(
            None, [mesh], "fast", False, render_mesh_cache={})

        assert calls["skeletonize"] == 1
        # The render-boundary product is a reduced TreeNeuron.
        assert all(isinstance(n, navis.TreeNeuron) for n in out)
        # Input mesh untouched.
        assert isinstance(mesh, navis.MeshNeuron)

    def test_tube_mode_never_skeletonizes(self, monkeypatch):
        visualizer, calls = build_line_visualizer(monkeypatch)
        visualizer.skeleton_mode = "tube"
        visualizer.skeleton_mesh_simplification = 0.95
        visualizer.soma_mesh_simplification = 0.80
        visualizer.soma_region_radius = 20.0
        mesh = make_mesh()

        out, done = visualizer._process_fafb_layer(
            None, [mesh], "fast", False, render_mesh_cache={})

        assert calls["skeletonize"] == 0
        assert isinstance(out[0], navis.MeshNeuron)

    def test_line_tree_source_never_skeletonizes(self, monkeypatch):
        visualizer, calls = build_line_visualizer(monkeypatch)
        neuron = make_neuron(2000)

        out, done = visualizer._process_fafb_layer(
            navis.NeuronList([neuron]), [], "fast", False,
            render_mesh_cache={})

        assert calls["skeletonize"] == 0
        assert isinstance(out[0], navis.TreeNeuron)
        # 90% line reduction applied to the skeleton source.
        assert out[0].n_nodes < neuron.n_nodes
        assert neuron.n_nodes == 2000  # canonical source untouched

    def test_skeletonized_product_never_written_to_cache(
            self, monkeypatch):
        visualizer, calls = build_line_visualizer(monkeypatch)
        mesh = make_mesh()

        # A cache write from the line branch would raise via the
        # fail_on_write stubs above.
        out, _ = visualizer._process_fafb_layer(
            None, [mesh], "fast", False, render_mesh_cache={})

        assert calls["skeletonize"] == 1
        assert all(isinstance(n, navis.TreeNeuron) for n in out)
