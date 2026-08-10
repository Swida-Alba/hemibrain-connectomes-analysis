"""Simplified NeuPrint skeleton cache (Mission 1.2).

Covers `VisualizeSkeleton._save_cached_neurons` / `_load_cached_neurons` /
`_skeleton_cache_is_simplified`:
- NeuPrint caches hold ONLY the fixed 90%-simplified skeleton
  (downsample factor 10) plus the `.level` marker; raw is never persisted
- a less-simplified render ignores the cache (`ignore_cache=True`) so the
  RAW skeletons are re-fetched transiently
- the FlyWire path is untouched (no downsample, no marker)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class FakeClient:
    """Minimal neuprint client stub: construction must not hit the network."""

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get("dataset", "hemibrain:v1.2.1")


def make_neuron(n_nodes=120):
    import navis
    # proper SWC classification (root/slab/end) so navis.downsample_neuron
    # reduces the skeleton like it does for real NeuPrint data
    types = (["root"] + ["slab"] * max(0, n_nodes - 2) + ["end"])
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n_nodes - 1)), dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * 1000.0,
        "y": np.zeros(n_nodes),
        "z": np.zeros(n_nodes),
        "radius": np.ones(n_nodes),
        "type": types[:n_nodes],
    })
    nrn = navis.TreeNeuron(nodes)
    # disable the constructor's soma auto-detection (it flags most nodes of
    # a plain chain, turning them into downsample fix points)
    nrn.soma = None
    return nrn


def build_vs(tmp_path, dataset="hemibrain:v1.2.1", cache_neurons=True):
    vs = VisualizeSkeleton(
        dataset=dataset,
        neuron_layers=["aMe12"],
        client=FakeClient(dataset=dataset),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
        cache_neurons=cache_neurons,
        data_folder=str(tmp_path),
        # sandbox the cache root (defaults to the project root)
        script_path=str(tmp_path),
    )
    vs.client_type = "neuprint"
    return vs


class TestNeuPrintSimplifiedCache:
    def test_save_writes_simplified_and_marker(self, tmp_path):
        vs = build_vs(tmp_path)
        neuron = make_neuron(120)
        neuron.id = 101
        df = pd.DataFrame({"bodyId": [101]})
        vs._save_cached_neurons(df, [neuron])

        cache_dir = Path(tmp_path) / "cache" / "hemibrain_v1_2_1" / "skeletons"
        pkl = cache_dir / "101.pkl"
        assert pkl.exists()
        assert (cache_dir / ".level").read_text().strip() == "simp90"
        with open(pkl, "rb") as f:
            cached = pickle.load(f)
        # simplified: strictly fewer nodes than the raw 120-node neuron
        assert len(cached.nodes) < len(neuron.nodes)

    def test_load_reads_simplified_cache(self, tmp_path):
        vs = build_vs(tmp_path)
        neuron = make_neuron(120)
        neuron.id = 101
        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])
        neurons, missing = vs._load_cached_neurons(pd.DataFrame({"bodyId": [101, 202]}))
        assert missing == [202]
        assert neurons is not None and len(neurons) == 1
        assert vs._skeleton_cache_is_simplified() is True

    def test_ignore_cache_reports_all_missing(self, tmp_path):
        vs = build_vs(tmp_path)
        neuron = make_neuron(120)
        neuron.id = 101
        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])
        # less-simplified render: the simp90 cache is too coarse -> refetch
        neurons, missing = vs._load_cached_neurons(
            pd.DataFrame({"bodyId": [101]}), ignore_cache=True)
        assert neurons is None
        assert missing == [101]

    def test_skeleton_cache_is_simplified_reads_marker(self, tmp_path):
        vs = build_vs(tmp_path)
        assert vs._skeleton_cache_is_simplified() is False  # no marker yet
        cache_dir = Path(tmp_path) / "cache" / "hemibrain_v1_2_1" / "skeletons"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / ".level").write_text("simp90\n")
        assert vs._skeleton_cache_is_simplified() is True


class TestFlywireCacheUntouched:
    def test_save_does_not_downsample_fafb(self, tmp_path, monkeypatch):
        """The FlyWire path keeps its own (mesh) cache: no downsample, no
        level marker, raw neurons stored as-is."""
        # the constructor checks the real FlyWire data files; skip that
        # check (the cache logic itself is what is under test)
        import FAFB_file_converter
        monkeypatch.setattr(FAFB_file_converter, "ensure_flywire_data",
                            lambda dataset, dataset_dir: True)
        vs = build_vs(tmp_path, dataset="flywire_FAFB_v783")
        vs.client_type = "flywire"
        neuron = make_neuron(120)
        neuron.id = 101
        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])

        cache_dir = Path(tmp_path) / "cache" / "flywire_FAFB_v783" / "skeletons"
        pkl = cache_dir / "101.pkl"
        assert pkl.exists()
        assert not (cache_dir / ".level").exists()
        with open(pkl, "rb") as f:
            cached = pickle.load(f)
        assert len(cached.nodes) == len(neuron.nodes)  # unchanged


class TestCustomLayerGrouping:
    """layer_map_csv: rows sharing a 'layer' value become ONE group (one
    legend entry in 'layer' mode, one individual profile)."""

    def test_layer_map_csv_groups_neurons(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "layer_map.csv"
        csv_path.write_text(
            "layer,id_type_instance\n"
            "DNp_group,DNp01\n"
            "DNp_group,DNp02\n"
            "aMe_group,aMe12\n"
        )
        # layer resolution happens via statvis during construction; the
        # grouping logic itself is what is under test
        monkeypatch.setattr(
            vs_mod.sv, "getNeurons",
            lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), "x", None),
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["aMe12"],  # overridden by the CSV
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        # two groups -> two layers, one per group
        assert vs.neuron_layers == [["DNp01", "DNp02"], "aMe12"]
        assert vs.custom_layer_names == ["DNp_group", "aMe_group"]
        assert vs.layer_names == ["DNp_group", "aMe_group"]

    def test_duplicate_custom_names_group_layers(self, tmp_path, monkeypatch):
        """The UI hint path: two layers given the SAME custom name share one
        legend entry in 'layer' mode (verified backend grouping)."""
        monkeypatch.setattr(
            vs_mod.sv, "getNeurons",
            lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), "x", None),
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["DNp01", "DNp02"],
            custom_layer_names=["DNp_group", "DNp_group"],
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        assert vs.layer_names == ["DNp_group", "DNp_group"]
        # in 'layer' mode both layers map to the same legend group
        assert len(set(vs.layer_names)) == 1
