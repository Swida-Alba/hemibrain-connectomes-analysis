"""Unit tests for the NeuPrint FAFB-format transform + transient mesh path.

Covers `VisualizeSkeleton` helpers added for the NeuPrint tube pipeline:
- `_resolved_skeleton_radius_style`
- `_fafb_style_radius` (tip taper, soma cap, branchpoint preservation)
- `_fafb_style_neuprint_skeleton` (smooth + resample + radius)
- the retired `_load_cached_neuprint_meshes` / `_save_cached_neuprint_meshes`
  compatibility seams
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
import visualize_skeleton as visualize_skeleton_module  # noqa: E402
import morphology as morph_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class FakeClient:
    """Minimal neuprint client stub: construction must not hit the network."""

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get("dataset", "hemibrain:v1.2.1")
        self.all_rois = set()


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


def batch_body_ids(batch):
    """Read IDs from either the legacy DataFrame or bound criteria input."""
    if isinstance(batch, pd.DataFrame):
        return batch["bodyId"].tolist()
    return np.asarray(batch.bodyId).tolist()


def build_vs(tmp_path, dataset="hemibrain:v1.2.1", pipeline=None):
    kwargs = dict(
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
    if pipeline is not None:
        kwargs["neuprint_skeleton_pipeline"] = pipeline
    vs = VisualizeSkeleton(**kwargs)
    vs.client_type = "neuprint"
    return vs


class TestNeuprintPipeline:
    def test_default_pipeline_is_fast(self, tmp_path):
        vs = build_vs(tmp_path)
        assert vs.neuprint_skeleton_pipeline == "fast"
        assert vs._resolved_neuprint_skeleton_pipeline() == "direct"

    def test_user_facing_pipeline_aliases_resolve(self, tmp_path):
        assert build_vs(tmp_path, pipeline="fast")._resolved_neuprint_skeleton_pipeline() == "direct"
        assert build_vs(tmp_path, pipeline="fine")._resolved_neuprint_skeleton_pipeline() == "fine"
        assert build_vs(tmp_path, pipeline="artistic")._resolved_neuprint_skeleton_pipeline() == "artistic"
        assert build_vs(tmp_path, pipeline="fine_opt")._resolved_neuprint_skeleton_pipeline() == "fine"
        assert build_vs(tmp_path, pipeline="fine_opt1")._resolved_neuprint_skeleton_pipeline() == "artistic"
        assert build_vs(tmp_path, pipeline="direct")._resolved_neuprint_skeleton_pipeline() == "direct"
        assert build_vs(tmp_path, pipeline="fafb")._resolved_neuprint_skeleton_pipeline() == "fine"

    def test_fine_methods_default_to_mesh_cache_level(self, tmp_path):
        assert build_vs(tmp_path / "fast", pipeline="fast").skeleton_mesh_simplification == pytest.approx(0.90)
        assert build_vs(tmp_path / "fine", pipeline="fine").skeleton_mesh_simplification == pytest.approx(0.95)
        assert build_vs(tmp_path / "artistic", pipeline="artistic").skeleton_mesh_simplification == pytest.approx(0.95)

    def test_fafb_pipeline_defaults_follow_selected_method(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            visualize_skeleton_module.FAFB_file_converter,
            "ensure_flywire_data",
            lambda *args, **kwargs: True,
        )
        assert build_vs(
            tmp_path / "fafb-fast",
            dataset="flywire_FAFB_v783",
            pipeline="fast",
        ).skeleton_mesh_simplification == pytest.approx(0.90)
        assert build_vs(
            tmp_path / "fafb-fine",
            dataset="flywire_FAFB_v783",
            pipeline="fine",
        ).skeleton_mesh_simplification == pytest.approx(0.95)

    def test_artistic_fetches_bounded_batches_in_requested_order(
            self, tmp_path, monkeypatch):
        vs = build_vs(tmp_path, pipeline="artistic")
        vs.NEUPRINT_FETCH_BATCH_SIZE = 2
        vs.NEUPRINT_FETCH_MAX_THREADS = 3
        calls = []

        def fake_fetch(batch, **kwargs):
            body_ids = batch_body_ids(batch)
            calls.append((body_ids, dict(kwargs)))
            neurons = []
            for body_id in body_ids:
                neuron = make_chain(n_nodes=5)
                neuron.id = body_id
                neurons.append(neuron)
            return navis.NeuronList(neurons)

        monkeypatch.setattr(
            visualize_skeleton_module.neu,
            "fetch_skeletons",
            fake_fetch,
        )
        fetched = vs._fetch_neuprint_skeletons_batched(
            [12211, 12517, 12737, 12211],
            {"with_synapses": False, "missing_swc": "warn"},
        )

        assert [body_id for body_id, _ in calls] == [
            [12211, 12517], [12737],
        ]
        assert [neuron.id for neuron in fetched] == [12211, 12517, 12737]
        assert all(kwargs["parallel"] is True for _, kwargs in calls)
        assert [kwargs["max_threads"] for _, kwargs in calls] == [2, 1]
        assert all(kwargs["client"] is vs.client for _, kwargs in calls)

    def test_batched_fetch_reports_progress(self, tmp_path, monkeypatch):
        vs = build_vs(tmp_path, pipeline="fine")
        vs.verbose = True
        vs.NEUPRINT_FETCH_BATCH_SIZE = 2
        progress_instances = []

        class FakeProgress:
            def __init__(self, *args, **kwargs):
                self.total = kwargs.get("total")
                self.updates = []
                self.descriptions = []
                self.postfixes = []
                self.closed = False
                progress_instances.append(self)

            def update(self, count):
                self.updates.append(count)

            def reset(self, total=None):
                self.total = total

            def set_description(self, value):
                self.descriptions.append(value)

            def set_postfix_str(self, value):
                self.postfixes.append(value)

            def close(self):
                self.closed = True

            @staticmethod
            def write(*args, **kwargs):
                return None

        def fake_fetch(batch, **kwargs):
            neurons = []
            for body_id in batch_body_ids(batch):
                neuron = make_chain(n_nodes=5)
                neuron.id = int(body_id)
                neurons.append(neuron)
            return navis.NeuronList(neurons)

        monkeypatch.setattr(visualize_skeleton_module.tqdm, "write", lambda *args, **kwargs: None, raising=False)
        monkeypatch.setattr(visualize_skeleton_module, "tqdm", FakeProgress)
        monkeypatch.setattr(
            visualize_skeleton_module.neu, "fetch_skeletons", fake_fetch)

        vs._fetch_neuprint_skeletons_batched(
            [12211, 12517, 12737],
            {"with_synapses": False, "missing_swc": "warn"},
        )

        assert len(progress_instances) == 1
        progress = progress_instances[0]
        assert progress.total == 3
        assert progress.updates == [2, 1, 1, 1, 1]
        assert progress.postfixes[:2] == ["batch 1/2", "batch 2/2"]
        assert progress.descriptions[0] == "Vectorizing fetched skeletons"
        assert progress.descriptions[-1] == "Vector cache complete"
        assert progress.closed is True

    def test_fast_first_render_persists_simplified_cache_and_decimates_in_memory(
            self, tmp_path, monkeypatch):
        vs = build_vs(tmp_path, pipeline="fast")
        vs.skeleton_mode = "tube"
        vs.skeleton_mesh_simplification = 0.9
        raw = make_chain(n_nodes=200)
        raw.id = 12211

        def fake_fetch(batch, **kwargs):
            result = raw.copy()
            result.id = 12211
            return navis.NeuronList([result])

        decimation_calls = []
        monkeypatch.setattr(
            visualize_skeleton_module.neu, "fetch_skeletons", fake_fetch)
        monkeypatch.setattr(
            vs,
            "_simplify_mesh_for_neuprint_pipeline",
            lambda *args: decimation_calls.append(True) or args[0],
        )

        prepared, _ = vs._prepare_neuprint_skeletons_for_render(
            [12211], False, False, {})

        cache_file = (
            Path(tmp_path) / "cache" / "hemibrain_v1_2_1" / "skeletons"
            / "raw_skeletons" / "12211.swc.zst"
        )
        assert cache_file.exists()
        cached = morph_mod._load_cached_skeleton_file(cache_file)
        # shared pipeline default: 90% simplified, level recorded in header
        assert cached._drocat_simplification == 90
        assert len(cached.nodes) < len(raw.nodes)
        assert 12211 in prepared
        # Fast is a render-time simplification mode: the cached skeleton is
        # reused while direct mesh decimation runs in memory.
        assert decimation_calls == [True]

    def test_render_preprocessing_aggregates_layers_before_fetch(
            self, tmp_path, monkeypatch):
        """All layer IDs enter one cache/fetch/preparation phase."""
        vs = build_vs(tmp_path, pipeline="fast")
        vs.skeleton_mode = "line"
        vs.cache_neurons = False
        vs.neuron_dfs = [
            pd.DataFrame({"bodyId": [101, 202]}),
            pd.DataFrame({"bodyId": [202, 303]}),
        ]
        calls = []

        def fake_fetch(batch, **kwargs):
            body_ids = batch_body_ids(batch)
            calls.append(body_ids)
            neurons = []
            for body_id in body_ids:
                neuron = make_chain(n_nodes=5)
                neuron.id = body_id
                neurons.append(neuron)
            return navis.NeuronList(neurons)

        monkeypatch.setattr(
            visualize_skeleton_module.neu,
            "fetch_skeletons",
            fake_fetch,
        )
        prepared, prepared_mesh = vs._prepare_neuprint_skeletons_for_render(
            vs._aggregate_neuprint_body_ids(),
            use_neuprint_fine_pipeline=False,
            use_neuprint_mesh_cache=False,
            neuprint_mesh_cache={},
        )

        assert calls == [[101, 202, 303]]
        assert list(prepared) == [101, 202, 303]
        assert prepared_mesh is False

    def test_render_preprocessing_retries_transport_error(
            self, tmp_path, monkeypatch):
        """A connection-level NeuPrint error does not abort every layer."""
        vs = build_vs(tmp_path, pipeline="fast")
        vs.skeleton_mode = "line"
        vs.cache_neurons = False
        attempts = []

        def flaky_fetch(body_ids, fetch_kwargs=None, persist=True):
            attempts.append(list(body_ids["bodyId"]))
            if len(attempts) == 1:
                # neuprint-python's wrapped requests error has no response
                # body and therefore omits the word "connection".
                raise RuntimeError(
                    "Error accessing POST "
                    "https://neuprint.janelia.org/api/custom/custom"
                )
            neuron = make_chain(n_nodes=5)
            neuron.id = 404
            return navis.NeuronList([neuron])

        monkeypatch.setattr(vs, "_fetch_neuprint_skeletons_batched", flaky_fetch)
        monkeypatch.setattr(visualize_skeleton_module.time, "sleep", lambda *_: None)

        prepared, prepared_mesh = vs._prepare_neuprint_skeletons_for_render(
            [404],
            use_neuprint_fine_pipeline=False,
            use_neuprint_mesh_cache=False,
            neuprint_mesh_cache={},
        )

        assert attempts == [[404], [404]]
        assert list(prepared) == [404]
        assert prepared_mesh is False

    def test_matching_default_client_is_retained(self, tmp_path, monkeypatch):
        """A matching process-wide client is passed explicitly to navis."""
        matching = FakeClient(dataset="hemibrain:v1.2.1")
        monkeypatch.setattr("neuprint.default_client", lambda: matching)
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["aMe12"],
            client=None,
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            cache_neurons=False,
            data_folder=str(tmp_path),
            script_path=str(tmp_path),
        )
        assert vs.client is matching

    def test_artistic_line_mode_still_forces_direct(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="artistic")
        vs.skeleton_mode = "line"
        assert vs._resolved_neuprint_skeleton_pipeline() == "direct"
        assert vs._uses_neuprint_fine_pipeline() is False
        assert vs._uses_neuprint_fine_optimized_pipeline() is False

    def test_artistic_dispatches_vertex_clustering(self, tmp_path, monkeypatch):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=10)
        artistic = build_vs(tmp_path / "artistic", pipeline="artistic")
        artistic.skeleton_mode = "tube"
        artistic.NEUPRINT_FINE_OPT1_CLUSTER_SEARCH_STEPS = 2
        clustered = artistic._simplify_mesh_vertex_clustering(mesh, 80)
        assert 0 < len(clustered.faces) < len(mesh.faces)

        calls = []
        monkeypatch.setattr(
            artistic,
            "_simplify_mesh_vertex_clustering",
            lambda *_args: calls.append("cluster") or mesh,
        )
        assert artistic._simplify_mesh_for_neuprint_pipeline(mesh, 80) is mesh
        assert calls == ["cluster"]

        fine = build_vs(tmp_path / "fine", pipeline="fine")
        monkeypatch.setattr(
            fine,
            "_simplify_mesh_open3d",
            lambda *_args: calls.append("qem") or mesh,
        )
        assert fine._simplify_mesh_for_neuprint_pipeline(mesh, 80) is mesh
        assert calls[-1] == "qem"

    def test_shared_online_fetch_ids_cover_all_methods_and_cache_rules(
            self, tmp_path):
        body_df = pd.DataFrame({"bodyId": [101, 202]})

        # Fast tube mode and a finer render target both reuse the same raw
        # SWC source; simplification level never changes online fetch IDs.
        fast = build_vs(tmp_path / "fast", pipeline="fast")
        fast.neuron_dfs = [body_df]
        fast.skeleton_mesh_simplification = 0.9
        cached = make_chain(n_nodes=20)
        cached.id = 101
        fast._save_cached_neurons(body_df.iloc[[0]], [cached])
        assert fast._get_neuprint_online_fetch_ids(
            use_neuprint_fine_pipeline=False,
            use_neuprint_mesh_cache=False,
            neuprint_mesh_cache={},
        ) == [202]
        assert fast._get_neuprint_online_fetch_ids(
            use_neuprint_fine_pipeline=False,
            use_neuprint_mesh_cache=False,
            neuprint_mesh_cache={},
        ) == [202]

        # Both fine method names use the shared path and therefore request
        # every body not already represented by the transformed mesh cache.
        for pipeline in ("fine", "artistic"):
            fine = build_vs(tmp_path / pipeline, pipeline=pipeline)
            fine.neuron_dfs = [body_df]
            assert fine._get_neuprint_online_fetch_ids(
                use_neuprint_fine_pipeline=True,
                use_neuprint_mesh_cache=False,
                neuprint_mesh_cache={},
            ) == [101, 202]

        # Line mode follows the same raw cache-aware path.
        line = build_vs(tmp_path / "line", pipeline="fast")
        line.skeleton_mode = "line"
        line.neuron_dfs = [body_df]
        line._save_cached_neurons(body_df.iloc[[0]], [cached])
        assert line._get_neuprint_online_fetch_ids(
            use_neuprint_fine_pipeline=False,
            use_neuprint_mesh_cache=False,
            neuprint_mesh_cache={},
        ) == [202]

    def test_line_mode_forces_direct_pipeline_and_skips_fine_path(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="fine")
        vs.skeleton_mode = "line"

        assert vs._resolved_neuprint_skeleton_pipeline() == "direct"
        assert vs._uses_neuprint_fine_pipeline() is False
        assert vs._uses_neuprint_fine_optimized_pipeline() is False

    def test_direct_pipeline_bypasses_transformed_mesh_cache(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="direct")
        vs.skeleton_mesh_simplification = 0.9
        mesh = TestNeuprintMeshCache()._make_mesh(101)

        vs._save_cached_neuprint_meshes({101: mesh})
        loaded, missing = vs._load_cached_neuprint_meshes([101])

        assert loaded == {}
        assert missing == [101]
        assert not (
            Path(tmp_path) / "cache" / "hemibrain_v1_2_1" / "skeletons"
            / "NEUPRINT_simp90"
        ).exists()

    def test_invalid_pipeline_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="neuprint_skeleton_pipeline"):
            build_vs(tmp_path, pipeline="unknown")


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

    def test_hides_nested_navis_progress_and_restores_setting(
            self, tmp_path, monkeypatch):
        vs = build_vs(tmp_path)
        vs.verbose = 'full'
        n = make_chain(n_nodes=20, spacing=100.0)
        previous = navis.config.pbar_hide
        observed = []

        def fake_smooth(neuron, **kwargs):
            observed.append(navis.config.pbar_hide)
            return neuron

        monkeypatch.setattr(
            visualize_skeleton_module.navis, 'smooth_skeleton', fake_smooth)
        vs._fafb_style_neuprint_skeleton(n)

        assert observed == [True]
        assert navis.config.pbar_hide is previous

    def test_radius_profile_applied(self, tmp_path):
        vs = build_vs(tmp_path)
        n = make_chain()
        out = vs._fafb_style_neuprint_skeleton(n)
        r = out.nodes['radius'].astype(float)
        assert r.max() > 32.0  # base multiplier applied
        assert r.min() < 32.0  # tip taper applied

    def test_fine_preserves_unoptimized_reference_profile(self, tmp_path):
        vs = build_vs(tmp_path / "fine", pipeline="fine")
        n = make_chain(n_nodes=80, spacing=100.0)

        reference = vs._fafb_style_neuprint_skeleton(n, optimized=False)
        fine = vs._fafb_style_neuprint_skeleton(n, optimized=True)

        assert len(fine.nodes) == len(reference.nodes)
        np.testing.assert_allclose(
            fine.nodes[["x", "y", "z", "radius"]].to_numpy(dtype=float),
            reference.nodes[["x", "y", "z", "radius"]].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_constant_radius_skips_profile_and_keeps_fine_path(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="fine")
        vs.skeleton_radius_style = "constant"
        n = make_chain(n_nodes=80, spacing=100.0)
        out = vs._fafb_style_neuprint_skeleton(n)
        assert len(out.nodes) > len(n.nodes)
        assert out.nodes["radius"].nunique() == 1
        assert out.nodes["radius"].iloc[0] == pytest.approx(
            vs.NEUPRINT_DEFAULT_RADIUS
        )


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
        vs = build_vs(tmp_path, pipeline="fine")
        vs.skeleton_mesh_simplification = 0.95
        vs._save_cached_neuprint_meshes({101: self._make_mesh(101)})
        loaded, missing = vs._load_cached_neuprint_meshes([101, 202])
        assert loaded == {}
        assert missing == [101, 202]
        assert not list(
            (Path(tmp_path) / "cache" / "hemibrain_v1_2_1" / "skeletons")
            .glob("NEUPRINT_simp95*/*.pkl")
        )

    def test_gated_below_cache_level(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="fine")
        vs.skeleton_mesh_simplification = 0.5
        loaded, missing = vs._load_cached_neuprint_meshes([101])
        assert loaded == {} and missing == [101]

    def test_gated_for_flywire_dataset(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="fine")
        vs.dataset = "flywire_FAFB_v783"
        vs.skeleton_mesh_simplification = 0.95
        vs._save_cached_neuprint_meshes({101: self._make_mesh(101)})
        loaded, missing = vs._load_cached_neuprint_meshes([101])
        assert loaded == {} and missing == [101]

    def test_cache_key(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="fine")
        assert vs._get_neuprint_mesh_cache_key() == "NEUPRINT_simp95"

    def test_artistic_cache_key_isolated_from_qem_cache(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="artistic")
        assert vs._get_neuprint_mesh_cache_key() == (
            "NEUPRINT_simp95_vertexcluster"
        )

    def test_constant_radius_uses_separate_cache_key(self, tmp_path):
        vs = build_vs(tmp_path, pipeline="fine")
        vs.skeleton_radius_style = "constant"
        assert vs._get_neuprint_mesh_cache_key() == "NEUPRINT_simp95_radiusconstant"
