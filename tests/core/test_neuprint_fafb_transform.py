"""Unit tests for the NeuPrint FAFB-format transform + mesh cache.

Covers `VisualizeSkeleton` helpers added for the NeuPrint tube pipeline:
- `_resolved_skeleton_radius_style`
- `_fafb_style_radius` (tip taper, soma cap, branchpoint preservation)
- `_fafb_style_neuprint_skeleton` (smooth + resample + radius)
- `_load_cached_neuprint_meshes` / `_save_cached_neuprint_meshes` gating
  and round-trip
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class FakeClient:
    """Minimal neuprint client stub: construction must not hit the network."""

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get("dataset", "hemibrain:v1.2.1")


def make_chain(n_nodes=60, spacing=100.0):
    """Linear chain: node 0 is the root, node n-1 is the single tip."""
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n_nodes - 1)), dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * spacing,
        "y": np.zeros(n_nodes),
        "z": np.zeros(n_nodes),
        "radius": np.full(n_nodes, 32.0),
    })
    nrn = navis.TreeNeuron(nodes)
    nrn.soma = None
    nrn.id = 42
    return nrn


def build_vs(tmp_path, dataset="hemibrain:v1.2.1"):
    vs = VisualizeSkeleton(
        dataset=dataset,
        neuron_layers=["aMe12"],
        client=FakeClient(dataset=dataset),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
        cache_neurons=True,
        data_folder=str(tmp_path),
        script_path=str(tmp_path),
    )
    vs.client_type = "neuprint"
    return vs


class TestRadiusStyleResolution:
    def test_neuprint_defaults_to_fafb(self, tmp_path):
        vs = build_vs(tmp_path)
        assert vs._resolved_skeleton_radius_style() == "fafb"

    def test_flywire_defaults_to_source(self, tmp_path):
        # the flywire constructor requires real FAFB data; the resolver only
        # reads self.dataset, so override the attribute on a neuprint instance
        vs = build_vs(tmp_path)
        vs.dataset = "flywire_FAFB_v783"
        assert vs._resolved_skeleton_radius_style() == "source"

    def test_explicit_override(self, tmp_path):
        vs = build_vs(tmp_path)
        vs.skeleton_radius_style = "source"
        assert vs._resolved_skeleton_radius_style() == "source"
        vs.skeleton_radius_style = "fafb"
        assert vs._resolved_skeleton_radius_style() == "fafb"


class TestFafbStyleRadius:
    def test_tip_taper_and_base_multiplier(self, tmp_path):
        vs = build_vs(tmp_path)
        n = make_chain()
        out = vs._fafb_style_radius(n)
        r = out.nodes['radius'].astype(float)
        # tip (last node): distance-to-tip 0 -> base * tip_taper
        assert r.iloc[-1] == pytest.approx(
            32.0 * vs.NEUPRINT_RADIUS_BASE_FACTOR
            * vs.NEUPRINT_RADIUS_TIP_TAPER, rel=1e-6)
        # root (far from tip): taper ~ 1 -> base * factor
        assert r.iloc[0] == pytest.approx(
            32.0 * vs.NEUPRINT_RADIUS_BASE_FACTOR, rel=1e-2)
        # monotonic taper toward the tip
        assert r.iloc[-1] < r.iloc[0]

    def test_soma_cap(self, tmp_path):
        vs = build_vs(tmp_path)
        n = make_chain()
        n.nodes.loc[0, 'radius'] = 5000.0  # huge soma at the root
        out = vs._fafb_style_radius(n)
        assert out.nodes['radius'].max() == pytest.approx(
            vs.NEUPRINT_RADIUS_SOMA_CAP, rel=1e-6)

    def test_branchpoint_bump_preserved(self, tmp_path):
        vs = build_vs(tmp_path)
        n = make_chain()
        # give node 10 a branchpoint-style radius bump (58.5 vs 32)
        n.nodes.loc[10, 'radius'] = 58.5
        out = vs._fafb_style_radius(n)
        r = out.nodes['radius'].astype(float)
        # ratio of bumped node to its neighbour is preserved
        bumped = r.iloc[10] / r.iloc[11]
        assert bumped == pytest.approx(58.5 / 32.0, rel=0.05)

    def test_input_not_mutated(self, tmp_path):
        vs = build_vs(tmp_path)
        n = make_chain()
        before = n.nodes['radius'].copy()
        vs._fafb_style_radius(n)
        assert (n.nodes['radius'] == before).all()


class TestFafbStyleNeuprintSkeleton:
    def test_resamples_to_finer_rings(self, tmp_path):
        vs = build_vs(tmp_path)
        n = make_chain(n_nodes=80, spacing=100.0)
        out = vs._fafb_style_neuprint_skeleton(n)
        # median edge was 100; resample target = 100 / 3.6 ~ 27.8
        assert len(out.nodes) > len(n.nodes)
        edges = vs._neuron_edge_lengths(out)
        assert np.median(edges) < 40.0

    def test_radius_profile_applied(self, tmp_path):
        vs = build_vs(tmp_path)
        n = make_chain()
        out = vs._fafb_style_neuprint_skeleton(n)
        r = out.nodes['radius'].astype(float)
        assert r.max() > 32.0  # base multiplier applied
        assert r.min() < 32.0  # tip taper applied


class TestNeuprintMeshCache:
    def _make_mesh(self, mesh_id):
        import trimesh
        tm = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        )
        mn = navis.MeshNeuron(tm)
        mn.id = mesh_id
        mn.name = str(mesh_id)
        return mn

    def test_save_and_load_roundtrip(self, tmp_path):
        vs = build_vs(tmp_path)
        vs.skeleton_mesh_simplification = 0.9
        vs._save_cached_neuprint_meshes({101: self._make_mesh(101)})
        loaded, missing = vs._load_cached_neuprint_meshes([101, 202])
        assert missing == [202]
        assert 101 in loaded
        assert loaded[101].id == 101

    def test_gated_below_cache_level(self, tmp_path):
        vs = build_vs(tmp_path)
        vs.skeleton_mesh_simplification = 0.5
        loaded, missing = vs._load_cached_neuprint_meshes([101])
        assert loaded == {} and missing == [101]

    def test_gated_for_flywire_dataset(self, tmp_path):
        vs = build_vs(tmp_path)
        vs.dataset = "flywire_FAFB_v783"
        vs.skeleton_mesh_simplification = 0.95
        vs._save_cached_neuprint_meshes({101: self._make_mesh(101)})
        loaded, missing = vs._load_cached_neuprint_meshes([101])
        assert loaded == {} and missing == [101]

    def test_cache_key(self, tmp_path):
        vs = build_vs(tmp_path)
        assert vs._get_neuprint_mesh_cache_key() == "NEUPRINT_simp90"
