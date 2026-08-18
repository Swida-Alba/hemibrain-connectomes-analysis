"""Tests for the FAFB pipeline ordering in `_process_fafb_layer`.

Asserts the stage order for each tube pipeline:
- fast: node reduction strictly before tube conversion and surface
  decimation; MeshNeuron sources skip the node stage
- fine: no node stage
- artistic: never combines with node reduction (vertex clustering only)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import visualize_skeleton as vs_mod  # noqa: E402
from skeleton_simplification import (  # noqa: E402
    FAFB_FAST_NODE_RETENTION,
    simplify_skeleton_nodes,
)
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


def make_big_mesh(body_id="99", subdivisions=3):
    """MeshNeuron with plenty of faces so decimation always fires."""
    import trimesh
    mesh = navis.MeshNeuron(
        trimesh.creation.icosphere(subdivisions=subdivisions))
    mesh.id = body_id
    return mesh


class PipelineRecorder:
    """Stubbed visualizer recording stage events and progress messages."""

    def __init__(self, pipeline, monkeypatch, mode="tube"):
        self.events = []
        self.messages = []
        self.pipeline = pipeline

        def record(event, *args):
            self.events.append((event,) + tuple(args))
            return args[0] if args else None

        self.visualizer = object.__new__(VisualizeSkeleton)
        self.visualizer.skeleton_mode = mode
        self.visualizer.skeleton_mesh_simplification = 0.95
        self.visualizer.soma_mesh_simplification = 0.80
        self.visualizer.soma_region_radius = 20.0
        self.visualizer._vprint = lambda msg, *a, **k: self.messages.append(
            msg)

        real_simplify = simplify_skeleton_nodes
        monkeypatch.setattr(
            vs_mod, "simplify_skeleton_nodes",
            _recorded(record, real_simplify))

        monkeypatch.setattr(
            navis.conversion, "tree2meshneuron",
            lambda neuron, tube_points=None: _tube(record, neuron))

        self.visualizer._simplify_mesh_fafb_fine = (
            lambda trimesh_obj, target_faces:
                _dec(record, "decimate", trimesh_obj, target_faces))
        self.visualizer._simplify_mesh_vertex_clustering = (
            lambda trimesh_obj, target_faces:
                _dec(record, "vertex_decimate", trimesh_obj, target_faces))

    def run(self, neuron_vols=None, cached_mesh_neurons=None,
            use_fafb_cache=False, pipeline=None):
        return self.visualizer._process_fafb_layer(
            neuron_vols,
            cached_mesh_neurons or [],
            pipeline or self.pipeline,
            use_fafb_cache,
            render_mesh_cache={},
        )


def _recorded(record, real_simplify):
    def wrapper(neuron, reduction, preserve_nodes=None):
        record("reduce", reduction)
        return real_simplify(neuron, reduction,
                             preserve_nodes=preserve_nodes)
    return wrapper


def _tube(record, neuron):
    record("tube")
    mesh = navis.MeshNeuron({
        "vertices": np.array([
            (0.0, 0.0, 0.0), (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0), (0.0, 0.0, 100.0),
        ]),
        "faces": np.array([
            (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
        ], dtype=np.int64),
    })
    mesh.id = getattr(neuron, "id", None)
    return mesh


def _dec(record, name, trimesh_obj, target_faces):
    record(name, target_faces)
    return trimesh_obj


class TestFastPipelineOrder:
    def test_node_reduction_runs_before_tube_and_decimation(
            self, monkeypatch):
        recorder = PipelineRecorder("fast", monkeypatch)
        neuron = make_neuron(2000)

        out, done = recorder.run(neuron_vols=navis.NeuronList([neuron]))

        names = [event[0] for event in recorder.events]
        assert names == ["reduce", "tube", "decimate"], names
        assert done is True
        # The retention constant (25%) is passed as its complement.
        assert recorder.events[0][1] == 1.0 - FAFB_FAST_NODE_RETENTION
        # The raw source is canonical: never mutated by the pipeline.
        assert neuron.n_nodes == 2000
        # Output is a tube mesh.
        assert isinstance(out[0], navis.MeshNeuron)
        assert "🔽 reduce nodes 42 2000→" in "".join(recorder.messages)

    def test_fine_has_no_node_stage(self, monkeypatch):
        recorder = PipelineRecorder("fine", monkeypatch)
        neuron = make_neuron(2000)

        out, done = recorder.run(neuron_vols=navis.NeuronList([neuron]))

        names = [event[0] for event in recorder.events]
        assert names == ["tube", "decimate"], names
        assert done is True
        assert neuron.n_nodes == 2000
        assert not any("reduce nodes" in msg for msg in recorder.messages)

    def test_artistic_never_combines_with_node_reduction(self, monkeypatch):
        recorder = PipelineRecorder("artistic", monkeypatch)
        neuron = make_neuron(2000)

        out, done = recorder.run(
            neuron_vols=navis.NeuronList([neuron]), pipeline="artistic")

        names = [event[0] for event in recorder.events]
        assert names == ["tube", "vertex_decimate"], names
        assert done is True
        assert not any("reduce nodes" in msg for msg in recorder.messages)

    def test_mesh_source_fast_skips_node_stage(self, monkeypatch):
        recorder = PipelineRecorder("fast", monkeypatch)
        mesh = make_big_mesh()

        out, done = recorder.run(cached_mesh_neurons=[mesh])

        names = [event[0] for event in recorder.events]
        assert "reduce" not in names
        assert "tube" not in names
        assert "decimate" in names
        assert done is True
        assert isinstance(out[0], navis.MeshNeuron)
        assert any("mesh source (no node stage)" in msg
                   for msg in recorder.messages)

    def test_fast_mesh_source_uses_relative_cache_step(self, monkeypatch):
        """Prepared-level meshes decimate with the relative keep factor,
        raw CAVE meshes with the absolute render target."""
        recorder = PipelineRecorder("fast", monkeypatch)
        mesh = make_big_mesh()
        n_faces = len(mesh.trimesh.faces)

        # prepared mesh cache eligible: relative step from 0.95 -> 0.98
        recorder.visualizer.skeleton_mesh_simplification = 0.98
        recorder.run(cached_mesh_neurons=[mesh], use_fafb_cache=True)
        cache_simp = recorder.visualizer.FAFB_MESH_CACHE_SIMPLIFICATION
        keep = (1 - 0.98) / (1 - cache_simp)
        expected = max(100, int(n_faces * keep))
        assert recorder.events[-1][1] == expected

        # cache bypassed: absolute target on the raw face count
        recorder.events.clear()
        recorder.visualizer.skeleton_mesh_simplification = 0.90
        recorder.run(cached_mesh_neurons=[mesh], use_fafb_cache=False)
        expected = max(100, int(n_faces * (1 - 0.90)))
        assert recorder.events[-1][1] == expected
